"""检查合成研究中的时间边界、账本和模型保存。
Check time boundaries, the synthetic ledger and model persistence.
"""
from dataclasses import replace
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efcuda.data import generate, make_features, make_labels, scale_panel, split_masks
from efcuda.ledger import backtest
from efcuda.engine import fit_network, predict
from efcuda.models import FullRankNet, MODEL_NAMES
from efcuda.cli import verify


def small_config():
    """内部测试用的小样本，直接调用模块函数以缩短运行时间。
    Small fixture for direct module tests; it does not pass through CLI validation.

这里的 24 只股票用于函数测试，不是命令行配置允许的最小股票数。
The 24-stock fixture is not the minimum universe size accepted by the CLI.
    """
    return {
        "scenario": "weak_signal", "seed": 17, "sessions": 330, "stocks": 24,
        "label_horizon": 5, "purge_sessions": 2, "train_end": 190,
        "validation_end": 240,
        "models": ["mlp", "gru", "gnn", "attention", "transformer"],
        "network": {"hidden": 8, "history": 8, "layers": 1, "heads": 2,
                    "dropout": 0.0, "lr": 0.001, "weight_decay": 0.001},
        "training": {"max_epochs": 2, "patience": 2, "batch_days": 2, "seed": 17},
        "portfolio": {"initial_capital": 1000000.0, "top_n": 5,
                      "rebalance_sessions": 20, "equity_fraction": 0.95,
                      "cost_bps": 10.0},
    }


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = small_config()
        cls.panel = generate(cls.config)

    def test_three_scenarios_are_reproducible_and_distinct(self):
        """同配置可重复，改变市场规则则价格序列不同。
        Identical settings reproduce prices; different market rules change them.
        """
        prices = []
        for name in ("weak_signal", "trend", "regime_shift"):
            config = dict(self.config, scenario=name)
            first, second = generate(config), generate(config)
            np.testing.assert_array_equal(first.opens, second.opens)
            np.testing.assert_array_equal(first.closes, second.closes)
            self.assertTrue(np.isfinite(first.opens).all())
            self.assertTrue((first.opens > 0).all())
            self.assertEqual(first.raw.shape[-1], 15)
            prices.append(first.closes)
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                self.assertFalse(np.array_equal(prices[i], prices[j]))

    def test_future_observations_do_not_change_past_features(self):
        """只改未来价格和成交量，过去的因子必须不变。
        Perturb future observations and require earlier features to stay unchanged.
        """
        closes, volume = self.panel.closes.copy(), self.panel.volume.copy()
        shares = np.full(closes.shape[1], 10000000.0)
        expected = make_features(closes, volume, shares)
        closes[200:] *= 1.8
        volume[200:] *= 3.0
        actual = make_features(closes, volume, shares)
        np.testing.assert_array_equal(expected[:200], actual[:200])
        self.assertFalse(np.array_equal(expected[210:], actual[210:]))

    def test_labels_use_next_open_entry_and_five_session_exit(self):
        """手算开盘到开盘收益，检查标签方向及最后六天缺失值。
        Check the open-to-open formula and the final six unavailable labels.
        """
        opens = np.arange(1, 61, dtype=float).reshape(20, 3) + 100.0
        raw, ranked = make_labels(opens, 5)
        expected = opens[6:] / opens[1:-5] - 1.0
        np.testing.assert_allclose(raw[:-6], expected, rtol=1e-7, atol=1e-8)
        self.assertTrue(np.isnan(raw[-6:]).all())
        self.assertTrue(np.isnan(ranked[-6:]).all())

    def test_purge_prevents_labels_crossing_split_boundaries(self):
        """以标签结束日检查隔离区，不能只看信号日是否在训练期。
        Check label exit dates against the purge boundary, not just signal dates.
        """
        parts = split_masks(self.panel, self.config)
        entry_to_exit = self.config["label_horizon"] + 1
        for key, boundary in (("train_mask", self.config["train_end"]),
                              ("valid_mask", self.config["validation_end"]),
                              ("final_mask", self.config["validation_end"])):
            days = np.flatnonzero(np.asarray(parts[key]).any(axis=1))
            self.assertGreater(len(days), 0)
            self.assertTrue(np.all(days + entry_to_exit < boundary - self.config["purge_sessions"]))
        self.assertGreaterEqual(int(parts["valid_days"][0]), self.config["train_end"])
        self.assertGreaterEqual(int(parts["test_days"][0]), self.config["validation_end"])

    def test_scaler_fits_training_observations_only(self):
        """训练样本外的极端值不得改变已估计的缩放参数。
        Extreme values outside training must not change fitted scaling parameters.
        """
        mask = split_masks(self.panel, self.config)["train_mask"]
        changed = self.panel.raw.copy()
        changed[~mask] = 100000.0
        _, original = scale_panel(self.panel, mask)
        _, altered = scale_panel(replace(self.panel, raw=changed), mask)
        self.assertEqual(set(original), set(altered))
        for key in original:
            np.testing.assert_array_equal(np.asarray(original[key]), np.asarray(altered[key]))


class LedgerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = generate(small_config())

    def flat_case(self, cost_bps):
        """固定价格以隔离费用影响，并构造固定的股票排名。
        Hold prices and rankings fixed to isolate transaction costs.
        """
        opens = np.full_like(self.panel.opens, 100.0)
        panel = replace(self.panel, opens=opens, closes=opens.copy())
        days = np.array([250, 251, 252, 253], dtype=int)
        scores = np.tile(np.arange(len(panel.codes), dtype=float), (len(days), 1))
        settings = dict(small_config()["portfolio"], cost_bps=cost_bps)
        return panel, days, scores, settings

    def test_constant_prices_lose_only_recorded_costs(self):
        """价格不动时，资产减少额应完全等于记录的交易费用。
        With flat prices, the decline in assets must equal recorded fees.
        """
        panel, days, scores, settings = self.flat_case(10.0)
        metrics, nav, trades = backtest(panel, days, scores, settings)
        self.assertGreater(metrics["fees"], 0)
        self.assertAlmostEqual(metrics["final_nav"], settings["initial_capital"] - metrics["fees"], places=6)
        self.assertAlmostEqual(metrics["total_return"], -metrics["fees"] / settings["initial_capital"], places=10)
        self.assertGreater(len(nav), 0)
        self.assertGreater(len(trades), 0)

    def test_signal_cannot_capture_price_move_before_next_open(self):
        """买入前的价格跳升不能计入策略收益。
        A price jump before the entry open must not become portfolio profit.
        """
        panel, days, scores, settings = self.flat_case(0.0)
        opens = panel.opens.copy()
        opens[:days[0] + 1] = 50.0
        panel = replace(panel, opens=opens, closes=opens.copy())
        metrics, _, _ = backtest(panel, days, scores, settings)
        self.assertAlmostEqual(metrics["final_nav"], settings["initial_capital"], places=6)

    def test_backtest_does_not_consume_future_labels(self):
        """替换未来标签后账本不变，证明此路径仅以预测评分选仓。
        Replacing future labels must leave this score-driven ledger unchanged.
        """
        panel, days, scores, settings = self.flat_case(10.0)
        original = backtest(panel, days, scores, settings)
        changed = replace(panel, target=np.full_like(panel.target, -999.0),
                          target_raw=np.full_like(panel.target_raw, 999.0))
        self.assertEqual(original, backtest(changed, days, scores, settings))


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        rng = np.random.default_rng(31)
        cls.panel = SimpleNamespace(
            x=rng.normal(size=(16, 24, 15)).astype(np.float32),
            target=rng.uniform(-0.5, 0.5, size=(16, 24)).astype(np.float32),
            present=np.ones((16, 24), dtype=bool),
            features=[f"feature_{i}" for i in range(15)], history=8,
        )
        cls.spec = small_config()["network"]
        cls.train_days, cls.valid_days = np.array([8, 9, 10]), np.array([11, 12])
        cls.train_mask = np.zeros((16, 24), dtype=bool)
        cls.valid_mask = np.zeros((16, 24), dtype=bool)
        cls.train_mask[cls.train_days] = True
        cls.valid_mask[cls.valid_days] = True

    def fit(self, panel, kind):
        return fit_network(
            panel, kind, self.spec, self.train_days, self.train_mask,
            self.valid_days, self.valid_mask, epochs=1, patience=1,
            seed=17, batch_days=2, device="cpu", execution="optimized",
        )

    def test_all_five_networks_fit_save_and_reinfer_on_cpu(self):
        """五类网络完成一次拟合，重载参数后重现同设备预测。
        Fit all five networks and reproduce same-device scores after weight reload.
        """
        for kind in MODEL_NAMES:
            with self.subTest(model=kind):
                model, epoch, rank_ic, curve = self.fit(self.panel, kind)
                scores = predict(model, self.panel, np.array([13, 14, 15]), batch_days=2)
                self.assertEqual(scores.shape, (3, 24))
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(np.isfinite(rank_ic))
                self.assertEqual(epoch, 1)
                self.assertEqual(len(curve), 1)
                self.assertEqual(model.execution_policy, "eager")
                stored = io.BytesIO()
                torch.save(model.state_dict(), stored)
                stored.seek(0)
                restored = FullRankNet(kind, 15, self.spec)
                restored.load_state_dict(torch.load(stored, map_location="cpu", weights_only=True))
                reloaded = predict(restored, self.panel, np.array([13, 14, 15]), batch_days=2)
                np.testing.assert_array_equal(scores, reloaded)

    def test_test_labels_do_not_change_training_weights(self):
        """污染测试标签后，固定种子的训练曲线与权重仍应一致。
        Altering test labels must leave fixed-seed training curves and weights unchanged.
        """
        targets = self.panel.target.copy()
        targets[13:] = 100000.0
        changed = SimpleNamespace(**{**vars(self.panel), "target": targets})
        original, _, _, first_curve = self.fit(self.panel, "mlp")
        altered, _, _, second_curve = self.fit(changed, "mlp")
        self.assertEqual(first_curve, second_curve)
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, altered.state_dict()[name]), name)


class IntegrityContractTests(unittest.TestCase):
    def test_changed_output_is_rejected(self):
        """模拟 CSV 被修改，确认文件哈希检查能够检出。
        Modify a CSV and require the file hash check to reject it.
        """
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as temporary:
            folder = Path(temporary)
            content = b"a,b\n1,2\n"
            (folder / "scores.csv").write_bytes(content)
            (folder / "manifest.json").write_text(json.dumps({"files": {
                "scores.csv": hashlib.sha256(content).hexdigest()}}), encoding="utf-8")
            self.assertEqual(verify(folder, "manifest.json")["verified_files"], 1)
            (folder / "scores.csv").write_bytes(content + b"3,4\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify(folder, "manifest.json")

    def test_manifest_cannot_read_outside_its_directory(self):
        """清单中的 ../ 路径应在访问目标文件之前被拒绝。
        Reject manifest traversal before reading a file outside the run directory.
        """
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as temporary:
            folder = Path(temporary)
            (folder / "manifest.json").write_text(json.dumps({"files": {
                "../outside.txt": "0" * 64}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify(folder, "manifest.json")


if __name__ == "__main__":
    unittest.main()
