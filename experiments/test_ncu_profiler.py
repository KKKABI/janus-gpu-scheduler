import unittest

from Opara.ncu_profiler import parse_ncu_csv


class ParseNcuCsvTests(unittest.TestCase):
    def test_parse_ncu_per_kernel_summary_csv(self):
        csv_text = '''==PROF== Connected
"Process ID","Kernel Name","Invocations","Section Name","Metric Name","Metric Unit","Minimum","Maximum","Average"
"1","kernel_a","2","GPU Speed Of Light Throughput","Memory Throughput","%","10","30","20"
"1","kernel_a","2","GPU Speed Of Light Throughput","DRAM Throughput","%","5","15","10"
"1","kernel_a","2","GPU Speed Of Light Throughput","L2 Cache Throughput","%","8","18","13"
"1","kernel_a","2","GPU Speed Of Light Throughput","Compute (SM) Throughput","%","12","22","17"
"1","kernel_a","2","GPU Speed Of Light Throughput","Duration","us","1","3","2"
'''

        self.assertEqual(parse_ncu_csv(csv_text), {
            "kernel_a": {
                "mem_thru": 20.0,
                "dram_thru": 10.0,
                "l2_thru": 13.0,
                "comp_thru": 17.0,
                "dur_ns": 2000.0,
            }
        })

    def test_parse_ncu_launch_csv_and_average_duplicate_rows(self):
        csv_text = '''"Kernel Name","Metric Name","Metric Value"
"kernel_a","Memory Throughput","10"
"kernel_a","Memory Throughput","30"
"kernel_a","Duration","4,096"
'''

        self.assertEqual(parse_ncu_csv(csv_text)["kernel_a"], {
            "mem_thru": 20.0,
            "dram_thru": 0.0,
            "l2_thru": 0.0,
            "comp_thru": 0.0,
            "dur_ns": 4096.0,
        })


if __name__ == "__main__":
    unittest.main()
