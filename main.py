"""
Agent 入口 — 支持 TUI 模式和命令行模式。
"""

import os
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 请设置 DEEPSEEK_API_KEY 环境变量或在 .env 文件中配置。")
        sys.exit(1)

    # --cli 参数走命令行模式，否则走 TUI
    if "--cli" in sys.argv:
        from agent import create_agent

        agent = create_agent(api_key=api_key)
        print("=" * 50)
        print("  LLM Agent — DeepSeek (命令行模式)")
        print("  输入 'exit' 或 'quit' 退出")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("再见！")
                break

            print("\nAgent 思考中...\n")

            def callback(event: str, data: dict) -> None:
                if event == "tool_call":
                    args_fmt = ", ".join(f"{k}={v}" for k, v in data["args"].items())
                    print(f"  🔧 调用工具: {data['name']}({args_fmt})")
                elif event == "tool_result":
                    print(f"  ↩ 结果: {data['result']}")

            try:
                answer = agent.run(user_input, callback=callback)
                print(f"Agent: {answer}")
            except Exception as e:
                print(f"错误: {e}")
    else:
        from tui import run_tui
        run_tui(api_key)


if __name__ == "__main__":
    main()
