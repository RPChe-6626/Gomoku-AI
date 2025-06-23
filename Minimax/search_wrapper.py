import subprocess
import os

class MinimaxPlayer:
    """
    一个Python包装器，用于与一个通过标准输入/输出进行通信的C++ AI可执行文件交互。
    """
    def __init__(self, board_width, board_height, exe_path):
        """
        初始化包装器。
        :param exe_path: C++ AI可执行文件的路径。
        """
        self.board_width = board_width
        self.board_height = board_height
        
        # 检查可执行文件是否存在
        if not os.path.exists(exe_path):
            raise FileNotFoundError(f"AI executable not found at: {exe_path}")
        self.exe_path = exe_path
        
        self.process = None
        self.player = None

    def _start_process(self):
        """启动C++子进程并进行初始化。"""
        if self.process:
            self.reset_player()

        # 启动子进程，并重定向其标准输入、输出和错误流
        self.process = subprocess.Popen(
            self.exe_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # 使用文本模式进行通信
            bufsize=1,  # 行缓冲
            universal_newlines=True
        )

        # 初始化C++ AI：发送它的执棋方
        # C++ AI: 0 for black, 1 for white
        # Python framework: 1 for black, 2 for white
        side = self.player - 1
        self._send(str(side))

        # 读取C++ AI返回的名字以完成初始化
        ai_name = self._read()
        print(f"Successfully started C++ AI: {ai_name}")

    def _send(self, message: str):
        """向子进程发送消息。"""
        if self.process and self.process.poll() is None:
            self.process.stdin.write(message + '\n')
            self.process.stdin.flush()
        else:
            print("Error: C++ process is not running.")

    def _read(self) -> str:
        """从子进程读取一行输出。"""
        if self.process and self.process.poll() is None:
            return self.process.stdout.readline().strip()
        else:
            print("Error: C++ process is not running.")
            return ""

    def set_player_ind(self, p):
        """实现接口1：设置玩家身份。"""
        self.player = p

    def reset_player(self):
        """实现接口2：重置玩家状态。通过终止并重启子进程来实现。"""
        if self.process:
            self.process.terminate()
            try:
                # 等待进程终止，设置超时以防死锁
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                print("Warning: C++ process did not terminate gracefully, killing it.")
                self.process.kill()
        self.process = None

    def get_action(self, board, is_selfplay=False, print_probs_value=0):
        """
        实现接口3：获取决策。
        这是与C++ AI交互的核心逻辑。
        """

        if self.player is None:
            current_player_on_board = board.get_current_player()
            self.set_player_ind(current_player_on_board)

        # 如果进程未启动（通常是游戏的第一步），则启动它
        if self.process is None:
            self._start_process()

        # 确定要发送给C++ AI的对手落子位置
        # C++ AI期望(-1, -1)表示它是第一个落子
        if board.last_move == -1:
            opponent_loc = (-1, -1)
        else:
            opponent_loc = board.move_to_location(board.last_move)

        # 将对手的落子位置发送给C++ AI
        self._send(f"{opponent_loc[0]} {opponent_loc[1]}")

        # 从C++ AI读取它计算出的落子位置
        response = self._read()
        if not response:
            print("Error: Did not receive a response from C++ AI.")
            # 作为一个备用方案，随机走一步
            return board.availables[0], None

        try:
            x_str, y_str = response.split()
            ai_loc = (int(x_str), int(y_str))
        except ValueError:
            print(f"Error: Could not parse response from C++ AI: '{response}'")
            return board.availables[0], None

        # 将C++ AI返回的坐标转换为Python框架使用的整数move
        move = board.location_to_move(ai_loc)

        # 按照接口要求返回move和None
        return move, None

    def __str__(self):
        return f"CPP_AI_Wrapper({self.exe_path})"

    def __del__(self):
        """确保在对象被销毁时，子进程也被终止。"""
        self.reset_player()