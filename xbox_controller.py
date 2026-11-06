import ctypes
import time
from ctypes import wintypes

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

PORT = "COM24"
ROBOT_ID = "my_awesome_follower_arm"

FPS = 30
DEADZONE = 0.12

BODY_SPEED = 14.0
WRIST_ROLL_SPEED = 20.0
GRIPPER_SPEED = 30.0

ERROR_SUCCESS = 0

XINPUT_GAMEPAD_A = 0x1000
XINPUT_GAMEPAD_B = 0x2000
XINPUT_GAMEPAD_X = 0x4000
XINPUT_GAMEPAD_Y = 0x8000
XINPUT_GAMEPAD_LEFT_SHOULDER = 0x0100
XINPUT_GAMEPAD_RIGHT_SHOULDER = 0x0200


class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", wintypes.WORD),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", wintypes.DWORD),
        ("Gamepad", XINPUT_GAMEPAD),
    ]


def load_xinput():
    for dll in ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"):
        try:
            lib = ctypes.WinDLL(dll)
            return lib, dll
        except OSError:
            pass
    raise RuntimeError("Could not load an XInput DLL.")


xinput, dll_name = load_xinput()
xinput.XInputGetState.argtypes = [wintypes.DWORD, ctypes.POINTER(XINPUT_STATE)]
xinput.XInputGetState.restype = wintypes.DWORD


def norm_stick(v: int) -> float:
    return max(-1.0, min(1.0, v / 32767.0))


def norm_trigger(v: int) -> float:
    return v / 255.0


def apply_deadzone(v: float) -> float:
    if abs(v) < DEADZONE:
        return 0.0
    sign = 1.0 if v >= 0 else -1.0
    return sign * (abs(v) - DEADZONE) / (1.0 - DEADZONE)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def find_controller():
    for idx in range(4):
        state = XINPUT_STATE()
        if xinput.XInputGetState(idx, ctypes.byref(state)) == ERROR_SUCCESS:
            return idx
    return None


def main():
    controller_index = find_controller()
    if controller_index is None:
        raise RuntimeError("No Xbox/XInput controller detected.")

    print(f"Using {dll_name}")
    print(f"Xbox controller found at XInput index {controller_index}")
    print()
    print("Controls:")
    print("  Left stick X     -> shoulder pan")
    print("  Left stick Y     -> shoulder lift")
    print("  Right stick Y    -> elbow flex")
    print("  Right stick X    -> wrist flex")
    print("  X / RB           -> wrist roll")
    print("  LT / RT          -> close / open gripper")
    print("  B                -> EXIT")
    print()
    print("Keep the arm clear. Test one control at a time.")

    config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
    )
    robot = SO101Follower(config)

    try:
        robot.connect()

        obs = robot.get_observation()
        target = {
            "shoulder_pan": float(obs["shoulder_pan.pos"]),
            "shoulder_lift": float(obs["shoulder_lift.pos"]),
            "elbow_flex": float(obs["elbow_flex.pos"]),
            "wrist_flex": float(obs["wrist_flex.pos"]),
            "wrist_roll": float(obs["wrist_roll.pos"]),
            "gripper": float(obs["gripper.pos"]),
        }

        print("\nFollower connected.")
        print("Starting from:")
        for joint, value in target.items():
            print(f"  {joint:14s}: {value:7.2f}")

        period = 1.0 / FPS
        last = time.perf_counter()

        while True:
            loop_start = time.perf_counter()
            dt = loop_start - last
            last = loop_start

            state = XINPUT_STATE()
            result = xinput.XInputGetState(controller_index, ctypes.byref(state))
            if result != ERROR_SUCCESS:
                raise RuntimeError("Xbox controller disconnected.")

            gp = state.Gamepad

            if gp.wButtons & XINPUT_GAMEPAD_B:
                print("B pressed. Exiting.")
                break

            lx = apply_deadzone(norm_stick(gp.sThumbLX))
            ly = apply_deadzone(norm_stick(gp.sThumbLY))
            rx = apply_deadzone(norm_stick(gp.sThumbRX))
            ry = apply_deadzone(norm_stick(gp.sThumbRY))
            lt = norm_trigger(gp.bLeftTrigger)
            rt = norm_trigger(gp.bRightTrigger)

            target["shoulder_pan"] += lx * BODY_SPEED * dt
            target["shoulder_lift"] += ly * BODY_SPEED * dt
            target["elbow_flex"] += ry * BODY_SPEED * dt
            target["wrist_flex"] += rx * BODY_SPEED * dt

            if gp.wButtons & XINPUT_GAMEPAD_X:
                target["wrist_roll"] -= WRIST_ROLL_SPEED * dt
            if gp.wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER:
                target["wrist_roll"] += WRIST_ROLL_SPEED * dt

            target["gripper"] += (rt - lt) * GRIPPER_SPEED * dt

            for joint in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ):
                target[joint] = clamp(target[joint], -100.0, 100.0)

            target["gripper"] = clamp(target["gripper"], 0.0, 100.0)

            action = {
                "shoulder_pan.pos": target["shoulder_pan"],
                "shoulder_lift.pos": target["shoulder_lift"],
                "elbow_flex.pos": target["elbow_flex"],
                "wrist_flex.pos": target["wrist_flex"],
                "wrist_roll.pos": target["wrist_roll"],
                "gripper.pos": target["gripper"],
            }

            robot.send_action(action)

            elapsed = time.perf_counter() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nCtrl+C received. Exiting.")

    finally:
        if robot.is_connected:
            robot.disconnect()
        print("Disconnected safely.")


if __name__ == "__main__":
    main()
