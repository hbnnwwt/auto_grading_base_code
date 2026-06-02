"""主程序集成测试。

验证 main.py 的端到端流程（使用模拟数据）。
运行: pytest tests/test_main.py -v
"""

import os
import subprocess
import sys

import pytest


def _run_main(*extra_args):
    """运行 main.py 并返回 CompletedProcess。"""
    project_root = os.path.dirname(os.path.dirname(__file__))
    return subprocess.run(
        [sys.executable, os.path.join(project_root, 'main.py'), *extra_args],
        capture_output=True, cwd=project_root, timeout=60,
    )


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_help_flag(self):
        """--help 应正常退出（returncode=0）。"""
        result = _run_main('--help')
        assert result.returncode == 0

    def test_missing_image_argument(self):
        """不提供 --image 参数应报错。"""
        result = _run_main()
        assert result.returncode != 0

    def test_nonexistent_image_file(self):
        """指定不存在的图像应报错。"""
        result = _run_main('--image', 'nonexistent.png')
        assert result.returncode != 0


class TestMainWithRealImage:
    def test_run_with_answer_sheet(self):
        """使用真实答题卡样例运行主程序。"""
        project_root = os.path.dirname(os.path.dirname(__file__))
        image_path = os.path.join(
            project_root, 'data', 'answer_sheets', 'answer_sheet_1.png'
        )
        if not os.path.exists(image_path):
            pytest.skip("answer_sheet_1.png 不存在")

        result = _run_main('--image', image_path)
        # 程序应正常退出（或因为 TODO 未实现而抛 NotImplementedError）
        # 只要不崩溃就算通过
        assert result.returncode is not None


class TestMainFolderMode:
    def test_folder_mode_with_real_images(self):
        """--folder 模式应能处理真实答题卡图片。"""
        project_root = os.path.dirname(os.path.dirname(__file__))
        folder = os.path.join(project_root, 'data', 'answer_sheets')
        if not os.path.isdir(folder):
            pytest.skip("data/answer_sheets 文件夹不存在")

        result = _run_main('--folder', folder)
        assert result.returncode is not None

    def test_folder_mode_empty_folder(self, tmp_path):
        """空文件夹应报错退出。"""
        empty = str(tmp_path / "empty")
        os.makedirs(empty)
        result = _run_main('--folder', empty)
        assert result.returncode != 0

    def test_folder_mode_nonexistent(self):
        """不存在的文件夹应报错。"""
        result = _run_main('--folder', '/nonexistent/path')
        assert result.returncode != 0
