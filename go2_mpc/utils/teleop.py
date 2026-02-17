import numpy as np
import threading
import sys


class KeyboardController:
    def __init__(self, vel_speed=0.1, yaw_speed=0.1):
        self.vel_speed = vel_speed
        self.yaw_speed = yaw_speed
        self.v_cmd_body = np.zeros(3)
        self.yaw_rate_cmd = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        """Background thread for keyboard input (terminal-based)."""
        import select
        import tty
        import termios

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key = sys.stdin.read(1).lower()
                    if key == 'w':
                        self.v_cmd_body[0] += self.vel_speed
                    elif key == 's':
                        self.v_cmd_body[0] -= self.vel_speed
                    elif key == 'a':
                        self.v_cmd_body[1] += self.vel_speed
                    elif key == 'd':
                        self.v_cmd_body[1] -= self.vel_speed
                    elif key == 'q':
                        self.yaw_rate_cmd += self.yaw_speed
                    elif key == 'e':
                        self.yaw_rate_cmd -= self.yaw_speed
                    elif key == ' ':
                        self.v_cmd_body[:] = 0.0
                        self.yaw_rate_cmd = 0.0
                    elif key == 'x':
                        self.running = False
        except Exception as e:
            print(f"Keyboard thread error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
