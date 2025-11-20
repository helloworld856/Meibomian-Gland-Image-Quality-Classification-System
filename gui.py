import sys
import os
from typing import List

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import predict
from predict import predict_single_image, predict_batch


class MainWindow(QMainWindow):
    """
    简单的图形界面：
    - 选择单张图片进行预测并展示结果
    - 选择文件夹进行批量预测并在文本框中显示结果
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像质量分类 - GUI")
        self.resize(900, 600)

        # 使用 predict 模块中已经加载好的模型
        try:
            self.model = predict.model
            if self.model is None:
                QMessageBox.warning(
                    self, 
                    "模型未加载", 
                    "模型加载失败，请确保模型文件存在。\n"
                    "预测功能将不可用。"
                )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "初始化错误", 
                f"初始化失败: {str(e)}\n"
                "请检查模型文件和配置文件。"
            )
            self.model = None

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 左侧：图片预览
        left_layout = QVBoxLayout()
        self.image_label = QLabel("图片预览")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet(
            "border: 1px solid #ccc; background-color: #f8f8f8;"
        )
        left_layout.addWidget(self.image_label)

        # 按钮区域
        button_layout = QHBoxLayout()
        self.btn_select_image = QPushButton("选择单张图片")
        self.btn_select_folder = QPushButton("选择图片文件夹")

        self.btn_select_image.clicked.connect(self.on_select_image)
        self.btn_select_folder.clicked.connect(self.on_select_folder)

        button_layout.addWidget(self.btn_select_image)
        button_layout.addWidget(self.btn_select_folder)

        left_layout.addLayout(button_layout)

        # 状态提示
        self.status_label = QLabel("")
        left_layout.addWidget(self.status_label)

        main_layout.addLayout(left_layout, stretch=1)

        # 右侧：结果显示
        right_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("预测结果将在这里显示...")
        right_layout.addWidget(QLabel("预测结果："))
        right_layout.addWidget(self.result_text, stretch=1)

        main_layout.addLayout(right_layout, stretch=1)

    # ========= 事件处理 =========
    def on_select_image(self):
        """选择单张图片并预测"""
        if self.model is None:
            QMessageBox.critical(self, "错误", "模型未加载，请检查模型文件。")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*.*)",
        )
        if not file_path:
            return

        self.show_image(file_path)
        self.status_label.setText("正在预测，请稍候...")
        QApplication.processEvents()

        try:
            results = predict_single_image(file_path, self.model, show_top_k=3)
            self.show_single_result(file_path, results)
            self.status_label.setText("预测完成")
        except Exception as e:
            QMessageBox.critical(self, "预测失败", str(e))
            self.status_label.setText("预测失败")

    def on_select_folder(self):
        """选择文件夹并进行批量预测"""
        if self.model is None:
            QMessageBox.critical(self, "错误", "模型未加载，请检查模型文件。")
            return

        folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "")
        if not folder_path:
            return

        self.image_label.setText("批量预测中...")
        self.status_label.setText("正在进行批量预测，请稍候...")
        QApplication.processEvents()

        try:
            results = predict_batch(folder_path, self.model, show_top_k=1)
            self.show_batch_result(folder_path, results)
            self.status_label.setText("批量预测完成")
        except Exception as e:
            QMessageBox.critical(self, "预测失败", str(e))
            self.status_label.setText("批量预测失败")

    # ========= 显示辅助函数 =========
    def show_image(self, file_path: str):
        """在左侧预览区域显示图片"""
        if not os.path.exists(file_path):
            self.image_label.setText("找不到图片")
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self.image_label.setText("无法加载图片")
            return

        # 等比缩放
        scaled = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def show_single_result(self, file_path: str, results: List[dict]):
        """将单张图片的预测结果显示在文本框中"""
        lines = [f"图片路径: {file_path}", ""]
        for i, item in enumerate(results, 1):
            lines.append(
                f"{i}. 类别: {item['class']}, 置信度: {item['confidence']:.4f} "
                f"({item['confidence'] * 100:.2f}%)"
            )
        self.result_text.setPlainText("\n".join(lines))

    def show_batch_result(self, folder_path: str, results: List[dict]):
        """将批量预测结果显示在文本框中"""
        lines = [f"文件夹: {folder_path}", f"共 {len(results)} 张图片", ""]
        for item in results:
            if "error" in item:
                lines.append(f"{item['filename']}: 错误 - {item['error']}")
            else:
                preds = item["predictions"]
                best = preds[0] if preds else None
                if best:
                    lines.append(
                        f"{item['filename']}: {best['class']} "
                        f"({best['confidence'] * 100:.2f}%)"
                    )
                else:
                    lines.append(f"{item['filename']}: 无预测结果")
        self.result_text.setPlainText("\n".join(lines))


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


