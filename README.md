# EDSR MindSpore Demo

1. [感谢](#感谢)

本项目 `model/edsr_x2.onnx` 由 [achie27/super-resolution](https://github.com/achie27/super-resolution.git) 项目提供的代码从 PyTorch 转换为 ONNX 得到.

本项目 `model/edsr_x2.ms` 由 MindSpore Lite 提供的 `converter` 工具从 ONNX 格式转换得到.

2. [运行](#运行)

运行本项目, 你需要安装 MindSpore lite. 请参考 [MindSpore 官网](https://www.mindspore.cn/lite) 安装 MindSpore lite.

然后复制 MindSpore Lite 中 `runtime/lib/mindspore-lite-java.jar` 到本项目的 `lib` 目录下. (如果没有请创建)

模型可以在 [Hugging Face 仓库](https://huggingface.co/JHT213/EDSR_MindSpore) 下载. 下载后将模型放到 `model` 目录下.(如果没有请创建)

最后运行 `src/main/java/com/example/Main.java` 即可.
