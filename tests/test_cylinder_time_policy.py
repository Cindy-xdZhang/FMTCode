import unittest
from FMT_Utils.CylinderTimePolicy import choose_cylinder_slice


class CylinderTimePolicyTests(unittest.TestCase):
    def test_original_range_and_earliest_cached_time(self):
        policy={'policy_version':'1.0','minimum_physical_start':7.,
                'minimum_original_simulation_fraction':.5,
                'simulation_time_ranges':{'cylinder3d':[0,15]},
                'cached_slice_selection':'earliest'}
        manifest={'source':{'time_min':0,'time_max':15},
                  'slices':[{'ordinal':i,'source_time':t} for i,t in enumerate([3.9,6.8,10.5,13.2])]}
        ordinal, audit=choose_cylinder_slice(manifest,'cylinder3d',policy)
        self.assertEqual(ordinal,2)
        self.assertEqual(audit['minimum_start_time'],7.5)
        manifest['source']['time_min']=7.5
        manifest['slices']=[{'ordinal':i,'source_time':t} for i,t in enumerate([9.5,10.8,12.1,13.4])]
        self.assertEqual(choose_cylinder_slice(manifest,'cylinder3d',policy)[0],0)

    def test_no_early_fallback(self):
        policy={'policy_version':'1.0','minimum_physical_start':7.,
                'minimum_original_simulation_fraction':.5,'simulation_time_ranges':{'c':[0,15]}}
        with self.assertRaises(ValueError):
            choose_cylinder_slice({'slices':[{'ordinal':0,'source_time':6.8}]},'c',policy)
