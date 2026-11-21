# video-watermark-detect

## 题面

版权检测接入台标识别后，同一台标报出二十个框、暗场还漏检：归一化减均值但除错（用均值），相关失真；NCC 不除模板能量，亮区永远高分；滑窗步长固定 1，大图慢 100 倍；不做 NMS，同一台标报 20 个框；投票按帧数而非权重，瞬时误检胜出；透明度估计用颜色比，忽略背景；模板大于图不报错，越界读；NaN 不校验，分数全 NaN；批量共用归一化缓冲，片之间串；threshold 超出 -1..1 不校验。logodetect/match.py 是核心，tests/test_match.py 15个红。要修的是：normalize 按总体标准差归一化并让常量图返回全零，ncc_at 除以图像与模板能量乘积的平方根，match 用与模板尺寸成比例的粗步长扫描并在候选附近细化，nms 按 IoU 抑制重叠框，temporal_vote 按帧去重并按置信度加权投票，estimate_opacity 用协方差除以模板方差估计透明度，validate 报告空图、NaN、模板越界等问题，detect_batch 隔离每一片并在任一片非法时整体回滚，match 与 detect_batch 校验 threshold 在 -1..1。

## 运行方式

```powershell
cd D:\gsb-20261119\video-watermark-detect
python -m pytest tests -q
```

初始状态应恰好输出：

```text
15 failed
```

15 条测试全部因 `logodetect/match.py` 中 8 个桩函数 `raise NotImplementedError` 而失败；收集、导入和语法检查必须通过，不能出现 `error`。修复核心模块后测试应全部通过。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_match.py` 是验收基准，不可修改、删除或新增；修复只能落在 `logodetect/match.py`。只使用 Python 标准库 `math` 与 pytest，离线运行，不使用随机数，也不 sleep。坐标单位为像素，灰度样本范围为 0..1，浮点断言容差为 1e-6。
