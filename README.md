# render-terrain-hole

## 题面

地形上挖了洞之后，洞口边缘一直有缝，远处的洞还会闪成黑块：地形按高度场挖洞时，要先用多边形遮罩标出洞，再找洞缘单元、生成缝合环、给洞内单元加深度偏移，最后按 LOD 保守降采样并裁剪与洞相交的三角形；hole_mask 边界点判成外部，洞口边缘少一圈；hole_mask 对自交与嵌套多边形用并集，奇偶规则错；boundary_cells 只查 4 邻域，斜向洞缘单元漏；stitch_ring 用集合去重，顶点绕序乱；stitch_ring 首尾不闭合，环上缺一段；depth_bias 偏移量取常量，近处洞穿帮；lod_consistent 直接用细遮罩，粗 LOD 漏洞；clip_triangles 整片删三角形，洞比预期大；validate_hole 不查边界闭合，开环洞不报；hole_batch 复用上一帧遮罩，帧间串洞。hole/punch.py 是核心，tests/test_punch.py 15个红。要修的是：按 8 个函数的 docstring 契约实现奇偶遮罩、8 邻域边界、稳定闭合的缝合环、相对洞缘的深度偏移、保守 LOD 降采样、三角形裁剪、洞完整性校验与逐帧隔离，让 15 个测试全部通过。

## 运行方式

在仓库根目录执行：

```powershell
cd D:\gsb-20261119\render-terrain-hole
python -m pytest tests -q
```

初始状态必须恰好是 `15 failed`，不能出现 collection error、import error 或 syntax error；每个失败都来自桩函数抛出的 `NotImplementedError`：

```text
15 failed in 0.XXs
```

修复后目标为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 文件结构

```text
render-terrain-hole/
  README.md  conftest.py  .gitignore
  hole/__init__.py  hole/punch.py
  tests/test_punch.py
```

`hole/punch.py` 的 8 个公开函数当前全部是 `raise NotImplementedError` 的桩；遮罩、边界单元、缝合环、深度偏移、LOD、三角形裁剪、完整性校验与批量处理的契约都写在函数 docstring 中，浮点比较统一使用绝对容差 `1e-9`。

## 约定

- `mask[y][x]` 与 `height[y][x]` 都是行优先二维列表，`x` 向右、`y` 向下；单元格 `(x, y)` 覆盖世界坐标方块 `[x, x + 1] x [y, y + 1]`。
- `height` 是地形高度场（米），`mask` 是挖洞遮罩（`True` 表示洞）。二者形状必须相同，高度必须有限。
- 缝合环节点写 `(x, y, height[y][x])`；裁剪结果写 `(x, y)` 浮点顶点，使用鞋带公式的有符号面积判断绕序。
- `hole_batch` 返回 `{"count": int, "frames": [...]}`，每帧包含 `id`、`bias`、`lod`、`issues`，顺序与输入一致。
- 只使用标准库 `math` 与 `pytest`，离线、无随机、不联网。

## 不可修改

`tests/test_punch.py` 是验收标准，不可修改；修复只能发生在 `hole/punch.py`，并保持 8 个公开函数的签名、返回结构与 docstring 契约不变。
