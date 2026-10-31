"""合成历史资料的来源身份校验；测试构造不代表真实研究结果。"""
from pathlib import Path
import json
import tempfile
import unittest

from eegseizguard.history import import_history
from eegseizguard.storage import file_hash, read_json, write_json


FINGERPRINTS = ('checkpoint_sha256', 'split_sha256', 'config_sha256',
                'windows_sha256', 'labels_summary_sha256', 'cache_manifest_sha256')


class HistoryIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.path = Path(self.directory.name)
        project = self.path / 'research'
        baseline = project / '05_raw_baseline/artifacts/r1_main/logreg_test_local'
        self.files = dict(predictions=baseline/'predictions.jsonl',
                          metadata=baseline/'predictions_meta.json',
                          metrics=baseline/'window_metrics.json',
                          events=project/'06_evaluation/artifacts/r1_main/logreg_test_local/metrics.json',
                          split=project/'04_patient_split/artifacts/r1_main/split.json')
        self.project = project
        baseline.mkdir(parents=True)
        rows = [dict(window_id=f'CHB-MIT:chb04_01:{i}', patient_id='CHB-MIT:chb04',
                     record_id='CHB-MIT:chb04_01', partition='test', y=y, prob=score,
                     end_seconds=40+i*10)
                for i, (y, score) in enumerate(((0, .1), (1, .9)))]
        self.files['predictions'].write_text('\n'.join(json.dumps(row) for row in rows)+'\n', 'utf8')
        write_json(self.files['split'], dict(test=['CHB-MIT:chb04']))
        identity = {key: f'{i+1:x}'*64 for i, key in enumerate(FINGERPRINTS)}
        identity['split_sha256'] = file_hash(self.files['split'])
        self.metadata = dict(partition='test', model='logreg', scientific_result=True,
                             predictions_sha256=file_hash(self.files['predictions']), **identity)
        self.events = dict(partition='test', model='logreg', scientific_result=True,
                           checkpoint_sha256=identity['checkpoint_sha256'],
                           identity={key: identity[key] for key in FINGERPRINTS[1:]},
                           event=dict(eligible_events=1, predicted_events=0,
                                      event_sensitivity=0., false_alarms=0, far_per_hour=0.),
                           alarm_policy=dict(strategy='never_alarm', selection_partition='validation'))
        write_json(self.files['metrics'], dict(partition='test', scientific_result=True,
                   macro_auroc=1., macro_auprc=1.,
                   per_patient={'CHB-MIT:chb04': dict(auroc=1., auprc=1.)}))
        self.save_identity()

    def tearDown(self):
        self.directory.cleanup()

    def save_identity(self):
        write_json(self.files['metadata'], self.metadata)
        write_json(self.files['events'], self.events)

    def event_fingerprint(self, key):
        return self.events if key == 'checkpoint_sha256' else self.events['identity']

    def test_complete_matching_identity_imports_without_changing_metrics(self):
        result = import_history(self.project, self.path/'valid')
        self.assertEqual(result['metrics']['macro']['auroc'], 1.)
        self.assertEqual(result['metrics']['macro']['ap'], 1.)
        self.assertTrue(result['source']['metric_verification']['per_patient_matches'])
        self.assertEqual(result['source']['historical_alarm_policy']['strategy'], 'never_alarm')

    def test_identity_missing_on_both_sides_is_not_a_match(self):
        for key in FINGERPRINTS:
            metadata, events = read_json(self.files['metadata']), read_json(self.files['events'])
            self.metadata.pop(key)
            self.event_fingerprint(key).pop(key)
            self.save_identity()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'SHA256|split'):
                import_history(self.project, self.path/('missing_'+key))
            self.metadata, self.events = metadata, events
            self.save_identity()

    def test_equal_malformed_fingerprints_are_rejected(self):
        for invalid in (None, '', 'not-a-digest', 'a'*63, True, 42):
            self.metadata['checkpoint_sha256'] = invalid
            self.events['checkpoint_sha256'] = invalid
            self.save_identity()
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, 'SHA256'):
                import_history(self.project, self.path/'invalid')

    def test_valid_but_mismatching_fingerprint_is_rejected(self):
        self.events['identity']['config_sha256'] = '0'*64
        self.save_identity()
        with self.assertRaisesRegex(ValueError, 'config_sha256.*不一致'):
            import_history(self.project, self.path/'mismatch')


if __name__ == '__main__':
    unittest.main()
