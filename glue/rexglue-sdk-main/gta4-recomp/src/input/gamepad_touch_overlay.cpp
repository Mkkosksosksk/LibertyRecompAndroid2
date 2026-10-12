#include "input/gamepad_touch_overlay.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_touch.h>

#include <rex/cvar.h>
#include <rex/input/absolute_pointer.h>
#include <rex/input/input.h>

REXCVAR_DEFINE_STRING(gta4_touch_layout, "gamepad", "GTA IV/Input/Touch",
                      "On-screen touch controls: gamepad (a fixed Xbox 360 controller) or context "
                      "(the context-sensitive layout)")
    .allowed({"gamepad", "context"});

REXCVAR_DEFINE_STRING(gta4_touch_style, "native", "GTA IV/Input/Touch",
                      "Touch button style: native (icons, CAR/WALK switch) or xbox (a fixed "
                      "Xbox 360 controller)")
    .allowed({"native", "xbox"});

REXCVAR_DEFINE_STRING(gta4_touch_camera, "drag", "GTA IV/Input/Touch",
                      "Camera control: drag (swipe the right side of the screen, like a native "
                      "mobile game) or stick (a floating right stick)")
    .allowed({"drag", "stick"});
REXCVAR_DEFINE_DOUBLE(gta4_touch_camera_sensitivity, 1.0, "GTA IV/Input/Touch",
                      "Camera swipe sensitivity")
    .range(0.25, 4.0);
REXCVAR_DEFINE_DOUBLE(gta4_touch_opacity, 0.8, "GTA IV/Input/Touch",
                      "Opacity of the on-screen controls")
    .range(0.2, 2.0);
REXCVAR_DEFINE_DOUBLE(gta4_touch_scale, 1.0, "GTA IV/Input/Touch",
                      "Size of the on-screen controls")
    .range(0.6, 1.6);

namespace gta4::input {
namespace {

enum class Kind : uint8_t { kStick, kButton, kTrigger, kArrow, kToggle };

enum class Icon : uint8_t {
  kNone, kSprint, kJump, kAim, kShoot, kDoor, kPhone, kFist, kCover,
  kGas, kBrake, kHandbrake, kHorn, kCamera, kLookBack, kArrowLeft, kArrowRight
};

struct Control {
  Kind kind = Kind::kButton;
  float x = 0.0f, y = 0.0f, r = 0.0f;  // display pixels; a stick's resting position
  uint16_t button = 0;                  // buttons, arrows
  uint8_t index = 0;                    // stick 0/1, trigger 0/1, arrow 0..3 (up, down, left, right)
  const char* label = "";
  ImU32 label_color = 0;                // 0: the common label colour
  Icon icon = Icon::kNone;              // drawn instead of the label when set
};

constexpr size_t kControlCount = 20;
constexpr size_t kLeftStick = 0;
constexpr size_t kRightStick = 1;
constexpr float kDeadZone = 0.12f;
constexpr float kCameraDragDeadZone = 0.02f;
// A button stays pressed at least this long, so a quick tap is not lost when
// the game renders slowly (the overlay is only sampled once per UI frame).
constexpr uint64_t kMinPressMs = 100;
constexpr uint64_t kTapExpiryMs = 250;
// A swipe of this many screen heights per second is a full camera deflection.
constexpr float kCameraFullSpeedHeights = 1.2f;

// Grey and see-through, so the overlay does not hide the game.
constexpr ImU32 kFill = IM_COL32(40, 40, 40, 75);
constexpr ImU32 kFillHeld = IM_COL32(180, 180, 180, 140);
constexpr ImU32 kRing = IM_COL32(245, 245, 245, 120);
constexpr ImU32 kLabel = IM_COL32(255, 255, 255, 200);
constexpr ImU32 kKnob = IM_COL32(240, 240, 240, 160);
constexpr ImU32 kIcon = IM_COL32(255, 255, 255, 230);

struct Finger {
  int control = -1;
};

struct State {
  std::mutex mutex;
  bool visible = false;
  std::array<Control, kControlCount> controls{};
  std::array<bool, kControlCount> held{};
  // Sticks float: the centre is where the finger came down.
  ImVec2 stick_origin[2] = {{0, 0}, {0, 0}};
  float stick_x[2] = {0, 0};  // -1..1, screen space (y down)
  float stick_y[2] = {0, 0};
  std::unordered_map<uint64_t, Finger> fingers;
  // Quick-tap latch (SDL_GetTicks, ms), per control.
  std::array<uint64_t, kControlCount> latch_until{};
  // Camera swipe: last finger position of each stick.
  ImVec2 stick_last[2] = {{0, 0}, {0, 0}};
  bool camera_drag = false;
  bool native_style = true;
  uint8_t mode = 0;  // native style: 0 = on foot, 1 = vehicle, 2 = cutscene
  float opacity = 1.0f;
  std::chrono::steady_clock::time_point last_update{};
};

State& state() {
  static State s;
  return s;
}

uint64_t FingerKey(SDL_TouchID touch, SDL_FingerID finger) {
  return (uint64_t(touch) << 32) ^ uint64_t(finger);
}

// Finger-down events are recorded as they happen (SDL may call this on another
// thread). A tap that starts and ends between two UI frames is never visible
// to SDL_GetTouchFingers, so it is turned into a short press from here.
struct Tap {
  uint64_t key;
  float x, y;  // normalised 0..1
  uint64_t ticks;
};
std::mutex g_tap_mutex;
std::vector<Tap> g_taps;
bool g_tap_watch_installed = false;

bool SDLCALL OnFingerEvent(void*, SDL_Event* event) {
  if (event->type == SDL_EVENT_FINGER_DOWN &&
      SDL_GetTouchDeviceType(event->tfinger.touchID) == SDL_TOUCH_DEVICE_DIRECT) {
    std::lock_guard lock(g_tap_mutex);
    if (g_taps.size() < 32) {
      g_taps.push_back({FingerKey(event->tfinger.touchID, event->tfinger.fingerID),
                        event->tfinger.x, event->tfinger.y, SDL_GetTicks()});
    }
  }
  return true;
}

ImU32 ScaleAlpha(ImU32 color, float factor) {
  const float alpha = float((color >> IM_COL32_A_SHIFT) & 0xFF) * factor;
  const uint32_t a = uint32_t(std::clamp(alpha, 0.0f, 255.0f));
  return (color & ~(0xFFu << IM_COL32_A_SHIFT)) | (a << IM_COL32_A_SHIFT);
}

// Sized from the screen height so the controls keep their physical size on
// 16:9 and 20:9 panels. The left half of the screen outside the buttons is the
// left stick (it centres wherever the finger lands); the right half is the
// right stick, i.e. the camera. The radar moves to the top-left corner while
// the overlay is up (ContextTouchHudLayoutActive).
std::array<Control, kControlCount> BuildLayout(float w, float h) {
  std::array<Control, kControlCount> c{};
  size_t n = 0;
  const auto add = [&](Kind kind, float x, float y, float r, uint16_t button, uint8_t index,
                       const char* label, ImU32 color = 0) {
    c[n++] = Control{kind, x, y, r, button, index, label, color};
  };
  const float lx = 0.12f * w;
  // Sticks first: kLeftStick, kRightStick. The right stick has no resting
  // place; it appears where the camera finger lands.
  add(Kind::kStick, lx, 0.72f * h, 0.105f * h, 0, 0, "L");
  add(Kind::kStick, w * 0.75f, 0.60f * h, 0.095f * h, 0, 1, "R");
  // D-pad, right of the left stick.
  const float dx = lx + 0.26f * h, dy = 0.86f * h, arm = 0.078f * h, dr = 0.042f * h;
  add(Kind::kArrow, dx, dy - arm, dr, rex::input::X_INPUT_GAMEPAD_DPAD_UP, 0, "");
  add(Kind::kArrow, dx, dy + arm, dr, rex::input::X_INPUT_GAMEPAD_DPAD_DOWN, 1, "");
  add(Kind::kArrow, dx - arm, dy, dr, rex::input::X_INPUT_GAMEPAD_DPAD_LEFT, 2, "");
  add(Kind::kArrow, dx + arm, dy, dr, rex::input::X_INPUT_GAMEPAD_DPAD_RIGHT, 3, "");
  // LT over LB above the left stick, L3 beside LB.
  const float br = 0.048f * h, small = 0.036f * h;
  add(Kind::kTrigger, lx, 0.355f * h, br, 0, 0, "LT");
  add(Kind::kButton, lx, 0.475f * h, br, rex::input::X_INPUT_GAMEPAD_LEFT_SHOULDER, 0, "LB");
  add(Kind::kButton, lx + 0.115f * h, 0.475f * h, small, rex::input::X_INPUT_GAMEPAD_LEFT_THUMB, 0,
      "L3");
  // A/B/X/Y diamond in the lower right, with the Xbox 360 letter colours.
  const float fx = w - 0.52f * h, fy = 0.80f * h, fs = 0.097f * h, fr = 0.052f * h;
  add(Kind::kButton, fx, fy + fs, fr, rex::input::X_INPUT_GAMEPAD_A, 0, "A",
      IM_COL32(110, 205, 80, 170));
  add(Kind::kButton, fx + fs, fy, fr, rex::input::X_INPUT_GAMEPAD_B, 0, "B",
      IM_COL32(235, 75, 70, 170));
  add(Kind::kButton, fx - fs, fy, fr, rex::input::X_INPUT_GAMEPAD_X, 0, "X",
      IM_COL32(75, 145, 240, 170));
  add(Kind::kButton, fx, fy - fs, fr, rex::input::X_INPUT_GAMEPAD_Y, 0, "Y",
      IM_COL32(245, 205, 50, 170));
  // RT over RB at the right edge, R3 beside RB.
  const float rx = w - 0.17f * h;
  add(Kind::kTrigger, rx, 0.47f * h, br, 0, 1, "RT");
  add(Kind::kButton, rx, 0.59f * h, br, rex::input::X_INPUT_GAMEPAD_RIGHT_SHOULDER, 0, "RB");
  add(Kind::kButton, rx - 0.115f * h, 0.59f * h, small, rex::input::X_INPUT_GAMEPAD_RIGHT_THUMB, 0,
      "R3");
  // Back and Start in the middle of the top edge.
  const float sr = 0.036f * h;
  add(Kind::kButton, 0.44f * w, 0.075f * h, sr, rex::input::X_INPUT_GAMEPAD_BACK, 0, "BACK");
  add(Kind::kButton, 0.56f * w, 0.075f * h, sr, rex::input::X_INPUT_GAMEPAD_START, 0, "START");
  return c;
}

// A layout in the style of a native mobile game: icons instead of Xbox letters,
// only the buttons that matter in the current situation, and a WALK / CAR / CUT
// switch at the top right. mode: 0 = on foot, 1 = in a vehicle, 2 = cutscene.
// The enter/exit-vehicle button (Y) sits in the same place in WALK and CAR mode,
// so it can never be replaced by another button.
std::array<Control, kControlCount> BuildNativeLayout(float w, float h, uint8_t mode) {
  std::array<Control, kControlCount> c{};
  size_t n = 0;
  const auto add = [&](Kind kind, float x, float y, float r, uint16_t button, uint8_t index,
                       const char* label, Icon icon = Icon::kNone, ImU32 color = 0) {
    if (n < c.size()) c[n++] = Control{kind, x, y, r, button, index, label, color, icon};
  };

  // Sticks first (kLeftStick, kRightStick): the left one floats under the thumb,
  // the right half of the screen is the camera.
  add(Kind::kStick, 0.12f * w, 0.72f * h, 0.105f * h, 0, 0, "L");
  add(Kind::kStick, w * 0.75f, 0.60f * h, 0.095f * h, 0, 1, "R");

  const float c1 = w - 0.12f * h;  // outer right column
  const float c2 = w - 0.32f * h;  // middle column
  const float c3 = w - 0.51f * h;  // inner column
  const ImU32 green_gas = IM_COL32(46, 204, 113, 220);
  const ImU32 red_brake = IM_COL32(231, 76, 60, 220);
  const ImU32 blue_action = IM_COL32(52, 152, 219, 200);

  // Top-right row, right to left, evenly spaced so the touch areas never overlap:
  // CUT, CAR, WALK, phone, weapon next, weapon previous.
  const float top_y = 0.11f * h, top_r = 0.045f * h, step = 0.125f * h;
  const auto top_x = [&](int k) { return w - 0.075f * h - float(k) * step; };
  add(Kind::kToggle, top_x(0), top_y, top_r, 0, 2, "CUT");
  add(Kind::kToggle, top_x(1), top_y, top_r, 0, 1, "CAR");
  add(Kind::kToggle, top_x(2), top_y, top_r, 0, 0, "WALK");
  add(Kind::kButton, top_x(3), top_y, top_r, rex::input::X_INPUT_GAMEPAD_DPAD_UP, 0, "", Icon::kPhone);
  add(Kind::kButton, top_x(4), top_y, top_r, rex::input::X_INPUT_GAMEPAD_DPAD_RIGHT, 0, "",
      Icon::kArrowRight);
  add(Kind::kButton, top_x(5), top_y, top_r, rex::input::X_INPUT_GAMEPAD_DPAD_LEFT, 0, "",
      Icon::kArrowLeft);
  // Top centre: BACK (map) and START (pause).
  add(Kind::kButton, 0.45f * w, 0.075f * h, 0.036f * h, rex::input::X_INPUT_GAMEPAD_BACK, 0, "BACK");
  add(Kind::kButton, 0.55f * w, 0.075f * h, 0.036f * h, rex::input::X_INPUT_GAMEPAD_START, 0, "START");

  if (mode == 0) {
    add(Kind::kButton, c1, 0.42f * h, 0.075f * h, rex::input::X_INPUT_GAMEPAD_Y, 0, "", Icon::kDoor,
        blue_action);  // enter vehicle
    add(Kind::kTrigger, c1, 0.65f * h, 0.085f * h, 0, 0, "", Icon::kAim);
    add(Kind::kTrigger, c2, 0.65f * h, 0.085f * h, 0, 1, "", Icon::kShoot, red_brake);
    add(Kind::kButton, c1, 0.86f * h, 0.090f * h, rex::input::X_INPUT_GAMEPAD_A, 0, "",
        Icon::kSprint, green_gas);
    add(Kind::kButton, c2, 0.86f * h, 0.075f * h, rex::input::X_INPUT_GAMEPAD_X, 0, "", Icon::kJump);
    add(Kind::kButton, c3, 0.80f * h, 0.065f * h, rex::input::X_INPUT_GAMEPAD_B, 0, "", Icon::kFist);
    add(Kind::kButton, c2, 0.42f * h, 0.060f * h, rex::input::X_INPUT_GAMEPAD_RIGHT_SHOULDER, 0, "",
        Icon::kCover);
  } else if (mode == 1) {
    add(Kind::kTrigger, c1, 0.82f * h, 0.105f * h, 0, 1, "", Icon::kGas, green_gas);
    add(Kind::kTrigger, c2, 0.84f * h, 0.090f * h, 0, 0, "", Icon::kBrake, red_brake);
    add(Kind::kButton, c1, 0.60f * h, 0.070f * h, rex::input::X_INPUT_GAMEPAD_RIGHT_SHOULDER, 0, "",
        Icon::kHandbrake);
    add(Kind::kButton, c1, 0.42f * h, 0.075f * h, rex::input::X_INPUT_GAMEPAD_Y, 0, "", Icon::kDoor,
        red_brake);  // exit vehicle
    add(Kind::kButton, c2, 0.60f * h, 0.055f * h, rex::input::X_INPUT_GAMEPAD_LEFT_THUMB, 0, "",
        Icon::kHorn);
    add(Kind::kButton, c3, 0.78f * h, 0.060f * h, rex::input::X_INPUT_GAMEPAD_B, 0, "", Icon::kCamera);
    add(Kind::kButton, c2, 0.42f * h, 0.060f * h, rex::input::X_INPUT_GAMEPAD_LEFT_SHOULDER, 0, "",
        Icon::kShoot);
    add(Kind::kButton, c3, 0.56f * h, 0.055f * h, rex::input::X_INPUT_GAMEPAD_RIGHT_THUMB, 0, "",
        Icon::kLookBack);
  } else {
    // Cutscene: the game shows which button skips it as an Xbox letter, so all four
    // face buttons and START are big and easy to hit.
    const float fx = w - 0.36f * h, fy = 0.70f * h, fs = 0.115f * h, fr = 0.062f * h;
    add(Kind::kButton, fx, fy + fs, fr, rex::input::X_INPUT_GAMEPAD_A, 0, "A",
        Icon::kNone, IM_COL32(110, 205, 80, 235));
    add(Kind::kButton, fx + fs, fy, fr, rex::input::X_INPUT_GAMEPAD_B, 0, "B",
        Icon::kNone, IM_COL32(235, 75, 70, 235));
    add(Kind::kButton, fx - fs, fy, fr, rex::input::X_INPUT_GAMEPAD_X, 0, "X",
        Icon::kNone, IM_COL32(75, 145, 240, 235));
    add(Kind::kButton, fx, fy - fs, fr, rex::input::X_INPUT_GAMEPAD_Y, 0, "Y",
        Icon::kNone, IM_COL32(245, 205, 50, 235));
    add(Kind::kButton, 0.5f * w, 0.84f * h, 0.075f * h, rex::input::X_INPUT_GAMEPAD_START, 0, "START");
  }
  return c;
}

// The button under a finger, or the stick of that half of the screen.
int HitTest(const std::array<Control, kControlCount>& controls, float px, float py, float w) {
  int best = -1;
  float best_distance = FLT_MAX;
  for (size_t i = 0; i < controls.size(); ++i) {
    const auto& control = controls[i];
    if (control.r <= 0.0f || control.kind == Kind::kStick) continue;
    const float capture = control.r * 1.35f;
    const float distance = std::hypot(px - control.x, py - control.y);
    if (distance <= capture && distance / capture < best_distance) {
      best_distance = distance / capture;
      best = int(i);
    }
  }
  if (best >= 0) return best;
  return int(px < w * 0.5f ? kLeftStick : kRightStick);
}

// The origin that draws a label's ink centred on `center`: horizontally by its
// advance, vertically by the glyph box of its first letter (labels are capitals),
// not by the line height, which leaves letters sitting high.
ImVec2 LabelOrigin(ImFont* font, float size, const char* label, ImVec2 center) {
  const ImVec2 text = font->CalcTextSizeA(size, FLT_MAX, 0.0f, label);
  float y = center.y - text.y * 0.5f;
  float x = center.x - text.x * 0.5f;
  if (ImFontBaked* baked = font->GetFontBaked(size)) {
    if (const ImFontGlyph* glyph = baked->FindGlyphNoFallback(ImWchar(label[0]))) {
      const float scale = baked->Size > 0.0f ? size / baked->Size : 1.0f;
      y = center.y - (glyph->Y0 + glyph->Y1) * 0.5f * scale;
      if (label[1] == 0) x = center.x - (glyph->X0 + glyph->X1) * 0.5f * scale;
    }
  }
  return ImVec2(x, y);
}

// Icons are drawn from lines and shapes, so there are no image files to ship.
// Coordinates are in units of ~0.8 of the button radius, x right, y down.
void DrawIcon(ImDrawList* d, Icon icon, ImVec2 c, float r, ImU32 col) {
  const float u = r * 0.8f;
  const float th = std::max(1.5f, u * 0.13f);
  const auto P = [&](float x, float y) { return ImVec2(c.x + x * u, c.y + y * u); };
  const auto line = [&](float x1, float y1, float x2, float y2) {
    d->AddLine(P(x1, y1), P(x2, y2), col, th);
  };
  const auto poly = [&](std::initializer_list<std::pair<float, float>> pts, bool closed) {
    ImVec2 v[16];
    int count = 0;
    for (const auto& p : pts) {
      if (count < 16) v[count++] = P(p.first, p.second);
    }
    d->AddPolyline(v, count, col, closed ? ImDrawFlags_Closed : ImDrawFlags_None, th);
  };
  const auto dot = [&](float x, float y, float radius) {
    d->AddCircleFilled(P(x, y), radius * u, col, 16);
  };

  switch (icon) {
    case Icon::kSprint:
      dot(0.22f, -0.62f, 0.17f);
      line(0.12f, -0.40f, -0.05f, 0.10f);
      poly({{0.10f, -0.30f}, {0.42f, -0.10f}, {0.62f, -0.32f}}, false);
      poly({{0.10f, -0.30f}, {-0.30f, -0.25f}, {-0.50f, 0.00f}}, false);
      poly({{-0.05f, 0.10f}, {0.30f, 0.30f}, {0.20f, 0.72f}}, false);
      poly({{-0.05f, 0.10f}, {-0.30f, 0.40f}, {-0.62f, 0.30f}}, false);
      break;
    case Icon::kJump:
      dot(0.0f, -0.65f, 0.17f);
      line(0.0f, -0.42f, 0.0f, 0.12f);
      line(0.0f, -0.32f, -0.42f, -0.70f);
      line(0.0f, -0.32f, 0.42f, -0.70f);
      poly({{0.0f, 0.12f}, {-0.32f, 0.42f}, {-0.22f, 0.78f}}, false);
      poly({{0.0f, 0.12f}, {0.32f, 0.42f}, {0.42f, 0.78f}}, false);
      break;
    case Icon::kAim:
      d->AddCircle(c, 0.50f * u, col, 32, th);
      line(0.0f, -0.78f, 0.0f, -0.30f);
      line(0.0f, 0.30f, 0.0f, 0.78f);
      line(-0.78f, 0.0f, -0.30f, 0.0f);
      line(0.30f, 0.0f, 0.78f, 0.0f);
      dot(0.0f, 0.0f, 0.07f);
      break;
    case Icon::kShoot: {
      // A bullet, tilted.
      const float ca = std::cos(0.7f), sa = std::sin(0.7f);
      const auto R = [&](float x, float y) {
        return ImVec2(c.x + (x * ca - y * sa) * u, c.y + (x * sa + y * ca) * u);
      };
      const ImVec2 body[] = {R(-0.20f, 0.55f), R(-0.20f, -0.15f), R(-0.10f, -0.45f), R(0.0f, -0.62f),
                             R(0.10f, -0.45f), R(0.20f, -0.15f), R(0.20f, 0.55f)};
      d->AddPolyline(body, 7, col, ImDrawFlags_Closed, th);
      d->AddLine(R(-0.20f, 0.33f), R(0.20f, 0.33f), col, th);
      break;
    }
    case Icon::kDoor:
      poly({{-0.50f, 0.65f}, {-0.50f, -0.05f}, {-0.22f, -0.58f}, {0.50f, -0.58f}, {0.50f, 0.65f}},
           true);
      poly({{-0.28f, -0.08f}, {-0.12f, -0.38f}, {0.32f, -0.38f}, {0.32f, -0.08f}}, true);
      line(0.0f, 0.12f, 0.30f, 0.12f);
      break;
    case Icon::kPhone:
      poly({{-0.32f, -0.70f}, {0.32f, -0.70f}, {0.32f, 0.70f}, {-0.32f, 0.70f}}, true);
      poly({{-0.20f, -0.50f}, {0.20f, -0.50f}, {0.20f, 0.30f}, {-0.20f, 0.30f}}, true);
      dot(0.0f, 0.52f, 0.07f);
      break;
    case Icon::kFist:
      poly({{-0.45f, -0.25f}, {0.45f, -0.25f}, {0.45f, 0.35f}, {0.20f, 0.62f}, {-0.35f, 0.62f},
            {-0.45f, 0.35f}},
           true);
      line(-0.15f, -0.25f, -0.15f, 0.05f);
      line(0.15f, -0.25f, 0.15f, 0.05f);
      line(-0.45f, 0.20f, -0.10f, 0.20f);
      break;
    case Icon::kCover:
      poly({{-0.48f, -0.58f}, {0.48f, -0.58f}, {0.48f, 0.05f}, {0.0f, 0.68f}, {-0.48f, 0.05f}},
           true);
      line(0.0f, -0.58f, 0.0f, 0.68f);
      break;
    case Icon::kGas:
      poly({{-0.28f, -0.68f}, {0.28f, -0.68f}, {0.28f, 0.68f}, {-0.28f, 0.68f}}, true);
      line(-0.18f, -0.35f, 0.18f, -0.35f);
      line(-0.18f, -0.05f, 0.18f, -0.05f);
      line(-0.18f, 0.25f, 0.18f, 0.25f);
      break;
    case Icon::kBrake:
      poly({{-0.50f, -0.32f}, {0.50f, -0.32f}, {0.50f, 0.38f}, {-0.50f, 0.38f}}, true);
      line(-0.30f, -0.12f, 0.30f, -0.12f);
      line(-0.30f, 0.12f, 0.30f, 0.12f);
      break;
    case Icon::kHandbrake:
      d->AddCircle(c, 0.60f * u, col, 32, th);
      line(0.0f, -0.30f, 0.0f, 0.10f);
      dot(0.0f, 0.34f, 0.07f);
      poly({{-0.72f, -0.35f}, {-0.84f, 0.0f}, {-0.72f, 0.35f}}, false);
      poly({{0.72f, -0.35f}, {0.84f, 0.0f}, {0.72f, 0.35f}}, false);
      break;
    case Icon::kHorn:
      poly({{-0.62f, -0.22f}, {-0.20f, -0.22f}, {0.38f, -0.58f}, {0.38f, 0.58f}, {-0.20f, 0.22f},
            {-0.62f, 0.22f}},
           true);
      poly({{0.58f, -0.30f}, {0.70f, 0.0f}, {0.58f, 0.30f}}, false);
      break;
    case Icon::kCamera:
      poly({{-0.60f, -0.30f}, {0.12f, -0.30f}, {0.12f, 0.40f}, {-0.60f, 0.40f}}, true);
      poly({{0.12f, 0.0f}, {0.62f, -0.28f}, {0.62f, 0.38f}, {0.12f, 0.12f}}, true);
      d->AddCircle(P(-0.35f, -0.52f), 0.16f * u, col, 16, th);
      break;
    case Icon::kLookBack: {
      // A circular arrow.
      ImVec2 v[16];
      int count = 0;
      float a = 0.0f;
      for (int k = 0; k < 15; ++k) {
        a = -0.9f + float(k) * 0.33f;
        v[count++] = P(0.55f * std::cos(a), 0.55f * std::sin(a));
      }
      d->AddPolyline(v, count, col, ImDrawFlags_None, th);
      const ImVec2 end = v[count - 1];
      const ImVec2 tangent(-std::sin(a), std::cos(a));
      const ImVec2 normal(std::cos(a), std::sin(a));
      d->AddTriangleFilled(ImVec2(end.x + tangent.x * 0.32f * u, end.y + tangent.y * 0.32f * u),
                           ImVec2(end.x + normal.x * 0.20f * u, end.y + normal.y * 0.20f * u),
                           ImVec2(end.x - normal.x * 0.20f * u, end.y - normal.y * 0.20f * u), col);
      break;
    }
    case Icon::kArrowLeft:
      d->AddTriangleFilled(P(-0.40f, 0.0f), P(0.30f, -0.50f), P(0.30f, 0.50f), col);
      break;
    case Icon::kArrowRight:
      d->AddTriangleFilled(P(0.40f, 0.0f), P(-0.30f, -0.50f), P(-0.30f, 0.50f), col);
      break;
    case Icon::kNone:
      break;
  }
}

void Draw(ImDrawList* draw, ImFont* font, float font_size, const State& s) {
  const auto A = [&](ImU32 color) { return ScaleAlpha(color, s.opacity); };
  for (size_t i = 0; i < s.controls.size(); ++i) {
    const auto& control = s.controls[i];
    if (control.r <= 0.0f) continue;
    const bool active = s.held[i] || (control.kind == Kind::kToggle && control.index == s.mode);
    const float radius = control.r;
    if (control.kind == Kind::kStick) {
      // The camera stick shows only while it is being used.
      if (control.index == 1 && (!active || s.camera_drag)) continue;
      const ImVec2 center = active ? s.stick_origin[control.index] : ImVec2(control.x, control.y);
      draw->AddCircleFilled(center, radius, A(kFill), 48);
      draw->AddCircle(center, radius, A(kRing), 48, std::max(1.0f, radius * 0.025f));
      const ImVec2 knob(center.x + s.stick_x[control.index] * radius * 0.55f,
                        center.y + s.stick_y[control.index] * radius * 0.55f);
      draw->AddCircleFilled(knob, radius * 0.34f, A(kKnob), 32);
      continue;
    }
    const ImVec2 center(control.x, control.y);
    draw->AddCircleFilled(center, radius, active ? A(kFillHeld) : A(kFill), 48);
    draw->AddCircle(center, radius, A(kRing), 48, std::max(1.0f, radius * 0.025f));
    if (control.icon != Icon::kNone) {
      DrawIcon(draw, control.icon, center, radius, A(control.label_color ? control.label_color : kIcon));
      continue;
    }
    if (control.kind == Kind::kArrow) {
      const float t = radius * 0.42f;
      ImVec2 a, b, c;
      switch (control.index) {
        case 0: a = {center.x, center.y - t}; b = {center.x - t, center.y + t * 0.6f}; c = {center.x + t, center.y + t * 0.6f}; break;
        case 1: a = {center.x, center.y + t}; b = {center.x + t, center.y - t * 0.6f}; c = {center.x - t, center.y - t * 0.6f}; break;
        case 2: a = {center.x - t, center.y}; b = {center.x + t * 0.6f, center.y + t}; c = {center.x + t * 0.6f, center.y - t}; break;
        default: a = {center.x + t, center.y}; b = {center.x - t * 0.6f, center.y - t}; c = {center.x - t * 0.6f, center.y + t}; break;
      }
      draw->AddTriangleFilled(a, b, c, A(kLabel));
      continue;
    }
    // Letters fill most of the circle; longer labels are fitted to its width.
    const ImVec2 natural = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, control.label);
    const float size = control.label[1] == 0 ? radius * 0.95f
        : natural.x > 0.0f ? std::min(radius * 0.62f, font_size * radius * 1.35f / natural.x)
                           : font_size;
    draw->AddText(font, size, LabelOrigin(font, size, control.label, center),
                  A(control.label_color ? control.label_color : kLabel), control.label);
  }
}

}  // namespace

bool GamepadTouchLayoutSelected() noexcept {
  return REXCVAR_GET(gta4_touch_layout) == "gamepad";
}

bool GamepadTouchOverlayVisible() noexcept {
  if (!GamepadTouchLayoutSelected()) return false;
  std::lock_guard lock(state().mutex);
  return state().visible;
}

bool UpdateAndDrawGamepadTouchOverlay(ImDrawList* draw, ImFont* font, float font_size,
                                      float width, float height) noexcept {
  auto& s = state();
  const bool visible = GamepadTouchLayoutSelected() && rex::input::TouchControlsVisible() &&
                       width > 0.0f && height > 0.0f;
  std::lock_guard lock(s.mutex);
  if (!visible) {
    s.visible = false;
    s.fingers.clear();
    s.held.fill(false);
    s.latch_until.fill(0);
    s.stick_x[0] = s.stick_x[1] = s.stick_y[0] = s.stick_y[1] = 0.0f;
    return false;
  }
  s.visible = true;
  s.native_style = REXCVAR_GET(gta4_touch_style) == "native";

  const float size_scale = float(std::clamp(rex::cvar::Query<double>("gta4_touch_scale"), 0.6, 1.6));
  const auto rebuild_layout = [&]() {
    s.controls = s.native_style ? BuildNativeLayout(width, height, s.mode)
                                : BuildLayout(width, height);
    for (auto& control : s.controls) control.r *= size_scale;
  };
  rebuild_layout();
  s.opacity = float(std::clamp(rex::cvar::Query<double>("gta4_touch_opacity"), 0.2, 2.0));
  s.camera_drag = REXCVAR_GET(gta4_touch_camera) == "drag";
  const float camera_sensitivity =
      float(std::clamp(rex::cvar::Query<double>("gta4_touch_camera_sensitivity"), 0.25, 4.0));

  const auto clock_now = std::chrono::steady_clock::now();
  float dt = s.last_update.time_since_epoch().count() == 0
                 ? 1.0f / 60.0f
                 : std::chrono::duration<float>(clock_now - s.last_update).count();
  s.last_update = clock_now;
  dt = std::clamp(dt, 1.0f / 240.0f, 0.1f);
  const uint64_t now = SDL_GetTicks();

  if (!g_tap_watch_installed) {
    SDL_AddEventWatch(OnFingerEvent, nullptr);
    g_tap_watch_installed = true;
  }

  // CAR/WALK: swaps the button cluster. Fingers resting on the old buttons are
  // dropped until they lift; the two sticks (indices 0 and 1) keep working.
  const auto set_mode = [&](uint8_t mode) {
    if (mode == s.mode) return;
    s.mode = mode;
    rebuild_layout();
    s.held.fill(false);
    s.latch_until.fill(0);
    for (auto& entry : s.fingers) {
      if (entry.second.control > int(kRightStick)) entry.second.control = -1;
    }
  };

  // Every finger currently on the screen, from every touch device.
  std::unordered_map<uint64_t, ImVec2> touches;
  int device_count = 0;
  if (SDL_TouchID* devices = SDL_GetTouchDevices(&device_count)) {
    for (int d = 0; d < device_count; ++d) {
      int finger_count = 0;
      if (SDL_Finger** fingers = SDL_GetTouchFingers(devices[d], &finger_count)) {
        for (int f = 0; f < finger_count; ++f) {
          const uint64_t key = FingerKey(devices[d], fingers[f]->id);
          touches[key] = ImVec2(fingers[f]->x * width, fingers[f]->y * height);
        }
        SDL_free(fingers);
      }
    }
    SDL_free(devices);
  }

  for (auto it = s.fingers.begin(); it != s.fingers.end();) {
    it = touches.count(it->first) ? std::next(it) : s.fingers.erase(it);
  }
  bool stick_taken[2] = {false, false};
  for (const auto& [key, finger] : s.fingers) {
    if (finger.control >= 0 && s.controls[finger.control].kind == Kind::kStick) {
      stick_taken[s.controls[finger.control].index] = true;
    }
  }

  s.held.fill(false);
  bool stick_owned[2] = {false, false};
  for (const auto& [key, position] : touches) {
    auto found = s.fingers.find(key);
    if (found == s.fingers.end()) {
      // A finger belongs to what it first landed on until it lifts. A stick
      // centres under the finger, kept fully on screen; one finger per stick.
      int control = HitTest(s.controls, position.x, position.y, width);
      if (control >= 0 && s.controls[control].kind == Kind::kStick) {
        const auto& stick = s.controls[control];
        if (stick_taken[stick.index]) {
          control = -1;
        } else {
          stick_taken[stick.index] = true;
          s.stick_origin[stick.index] =
              ImVec2(std::clamp(position.x, stick.r, width - stick.r),
                     std::clamp(position.y, stick.r, height - stick.r));
        }
      }
      if (control >= 0 && s.controls[control].kind == Kind::kToggle) {
        set_mode(s.controls[control].index);
        control = -1;  // this touch only switches the layout
      }
      if (control >= 0) {
        const auto& landed = s.controls[control];
        if (landed.kind == Kind::kStick) {
          s.stick_last[landed.index] = position;
        } else {
          s.latch_until[control] = now + kMinPressMs;
        }
      }
      found = s.fingers.emplace(key, Finger{control}).first;
    }
    const int control_index = found->second.control;
    if (control_index < 0) continue;
    const Control& control = s.controls[control_index];
    s.held[control_index] = true;
    if (control.kind != Kind::kStick) continue;
    float x, y;
    if (control.index == 1 && s.camera_drag) {
      // Native-style look: the camera follows how fast the finger moves, not
      // how far it is from where it landed, so the thumb never runs out of room.
      const ImVec2 last = s.stick_last[1];
      s.stick_last[1] = position;
      const float to_stick = camera_sensitivity / (kCameraFullSpeedHeights * height);
      const float target_x = (position.x - last.x) / dt * to_stick;
      const float target_y = (position.y - last.y) / dt * to_stick;
      x = s.stick_x[1] + (target_x - s.stick_x[1]) * 0.6f;
      y = s.stick_y[1] + (target_y - s.stick_y[1]) * 0.6f;
    } else {
      const ImVec2 origin = s.stick_origin[control.index];
      x = (position.x - origin.x) / (control.r * 0.85f);
      y = (position.y - origin.y) / (control.r * 0.85f);
    }
    const float length = std::hypot(x, y);
    if (length > 1.0f) {
      x /= length;
      y /= length;
    }
    s.stick_x[control.index] = x;
    s.stick_y[control.index] = y;
    stick_owned[control.index] = true;
  }
  for (int i = 0; i < 2; ++i) {
    if (!stick_owned[i]) s.stick_x[i] = s.stick_y[i] = 0.0f;
  }

  // Taps that began and ended since the last frame never showed up above.
  std::vector<Tap> taps;
  {
    std::lock_guard tap_lock(g_tap_mutex);
    taps.swap(g_taps);
  }
  for (const Tap& tap : taps) {
    if (touches.count(tap.key) || now - tap.ticks > kTapExpiryMs) continue;
    const int control = HitTest(s.controls, tap.x * width, tap.y * height, width);
    if (control < 0 || s.controls[control].kind == Kind::kStick) continue;
    if (s.controls[control].kind == Kind::kToggle) {
      set_mode(s.controls[control].index);
    } else {
      s.latch_until[control] = now + kMinPressMs;
    }
  }
  for (size_t i = 0; i < s.controls.size(); ++i) {
    if (s.controls[i].kind != Kind::kStick && s.controls[i].kind != Kind::kToggle &&
        now < s.latch_until[i])
      s.held[i] = true;
  }

  if (draw && font && font_size > 0.0f) Draw(draw, font, font_size, s);
  return true;
}

bool ReadGamepadTouchOverlay(uint32_t user, rex::input::X_INPUT_GAMEPAD* output) noexcept {
  if (!output || user != 0) return false;
  auto& s = state();
  std::lock_guard lock(s.mutex);
  if (!s.visible) return false;
  uint16_t buttons = 0;
  uint8_t triggers[2] = {0, 0};
  for (size_t i = 0; i < s.controls.size(); ++i) {
    const auto& control = s.controls[i];
    if (!s.held[i]) continue;
    if (control.kind == Kind::kTrigger) {
      triggers[control.index] = 255;
    } else if (control.kind != Kind::kStick && control.kind != Kind::kToggle) {
      buttons |= control.button;
    }
  }
  // Radial dead zone per stick; screen Y points down, the pad's up.
  const auto stick = [&](int i, int16_t* x_out, int16_t* y_out) {
    const float x = s.stick_x[i], y = s.stick_y[i];
    const float length = std::hypot(x, y);
    const float dead = (i == 1 && s.camera_drag) ? kCameraDragDeadZone : kDeadZone;
    if (length < dead) {
      *x_out = *y_out = 0;
      return;
    }
    const float scale = std::min((length - dead) / (1.0f - dead), 1.0f) / length;
    *x_out = int16_t(std::clamp(x * scale, -1.0f, 1.0f) * 32767.0f);
    *y_out = int16_t(std::clamp(-y * scale, -1.0f, 1.0f) * 32767.0f);
  };
  int16_t lx = 0, ly = 0, rx = 0, ry = 0;
  stick(0, &lx, &ly);
  stick(1, &rx, &ry);
  *output = {};
  output->buttons = buttons;
  output->left_trigger = triggers[0];
  output->right_trigger = triggers[1];
  output->thumb_lx = lx;
  output->thumb_ly = ly;
  output->thumb_rx = rx;
  output->thumb_ry = ry;
  return true;
}

}  // namespace gta4::input
