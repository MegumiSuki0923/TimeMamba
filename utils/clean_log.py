import os
import sys

# 清洗训练日志：模拟终端渲染，使 cleaned 文件和屏幕上打印的长一个样。
# 真实日志行以 \n 结尾；tqdm 进度条每次刷新写 \r + 内容（回车覆盖、不换行），
# 因此每个物理行只保留最后一个 \r 之后的画面——即屏幕上最终显示的内容，
# 进度条只剩最后定格帧，被覆盖的中间刷新帧（含验证条）自然消失。

for path in sys.argv[1:]:
    # newline='' 关闭 universal newlines，否则读入时 \r 会被转成 \n，覆盖语义就丢了
    with open(path, newline='') as f:
        text = f.read()

    raw_lines = text.split('\n')
    cleaned = [line.split('\r')[-1] for line in raw_lines]

    root, ext = os.path.splitext(path)
    out_path = f'{root}_cleaned{ext}'
    with open(out_path, 'w', newline='') as f:
        f.write('\n'.join(cleaned))

    print(f'{path} -> {out_path} ({len(raw_lines)} 行 -> {len(cleaned)} 行)')
