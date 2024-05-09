import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, \
    QFileDialog, QLineEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from test import Photo2Cartoon
import cv2

class ImageConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('人像卡通化')
        self.setGeometry(300, 300, 600, 400)

        # Layouts
        layout = QVBoxLayout()
        h_layout1 = QHBoxLayout()
        h_layout2 = QHBoxLayout()
        h_layout_images = QHBoxLayout()  # 用于放置两个垂直布局的水平布局

        # 垂直布局来放置原始图片和其标签
        v_layout_original = QVBoxLayout()
        self.original_label = QLabel('原始图片:')
        self.original_image_label = QLabel(self)
        v_layout_original.addWidget(self.original_label)
        v_layout_original.addWidget(self.original_image_label)

        # 垂直布局来放置转换后的图片和其标签
        v_layout_output = QVBoxLayout()
        self.output_label = QLabel('转换后的图片:')
        self.output_image_label = QLabel(self)
        v_layout_output.addWidget(self.output_label)
        v_layout_output.addWidget(self.output_image_label)

        # Widgets for input and buttons
        self.input_label = QLabel('请选择输入图片:')
        self.input_path = QLineEdit(self)
        self.browse_button = QPushButton('浏览', self)
        self.browse_button.clicked.connect(self.browse_input_image)
        self.convert_button = QPushButton('转换', self)
        self.convert_button.clicked.connect(self.convert_image)
        self.save_button = QPushButton('保存', self)
        self.save_button.clicked.connect(self.save_output_image)

        # Layout Arrangements
        h_layout1.addWidget(self.input_label)
        h_layout1.addWidget(self.input_path)
        h_layout1.addWidget(self.browse_button)

        h_layout2.addWidget(self.convert_button)
        h_layout2.addWidget(self.save_button)

        # 将两个垂直布局添加到水平布局中
        h_layout_images.addLayout(v_layout_original)
        h_layout_images.addLayout(v_layout_output)

        layout.addLayout(h_layout1)  # 输入部分
        layout.addLayout(h_layout_images)  # 图片部分
        layout.addLayout(h_layout2)  # 按钮部分

        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.show()

    def browse_input_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', 'Image files(*.png *.jpg *.bmp)')
        self.input_path.setText(fname)
        # 读取原始图片并在QLabel中显示
        self.show_original_image(fname)

    def show_original_image(self, image_path):
        # 读取原始图片并转换为QPixmap
        original_pixmap = QPixmap(image_path)
        self.original_image_label.setPixmap(original_pixmap.scaled(256, 256, Qt.KeepAspectRatio))

    def convert_image(self):
        # Add your image conversion logic here
        input_image_path = self.input_path.text()
        if not input_image_path:
            print("请选择输入图片！")
            return

            # 读取图片
        # img = cv2.imread(input_image_path)
        img = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
        if img is None:
            print("无法读取图片！")
            return

            # 创建Photo2Cartoon实例并进行图片转换
        c2p = Photo2Cartoon()
        cartoon = c2p.inference(img)
        if cartoon is None:
            print("无法检测到人脸，无法进行转换！")
            return

            # 将OpenCV图像转换为QImage，再转换为QPixmap以在QLabel中显示
        height, width, channel = cartoon.shape
        bytes_per_line = channel * width
        qimage = QImage(cartoon.data, width, height, bytes_per_line, QImage.Format_BGR888)#.rgbSwapped()
        self.output_pixmap = QPixmap.fromImage(qimage)

        # 显示转换后的图片
        self.output_image_label.setPixmap(self.output_pixmap.scaled(256,256, Qt.KeepAspectRatio))

    def save_output_image(self):
        if hasattr(self, 'output_pixmap'):
            fname, _ = QFileDialog.getSaveFileName(self, 'Save file', '.', 'Image files(*.png *.jpg *.bmp)')
            if fname:
                self.output_pixmap.save(fname)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageConverter()
    sys.exit(app.exec_())