"""Regression checks for Docker's unordered Mounts array; no Spark or Docker calls."""
import copy
import unittest

from studies.xgb_feasibility_v1.run import stable_metadata


class MetadataTests(unittest.TestCase):
    def test_mount_order_does_not_imply_mutation(self):
        original = {'Id':'one','Image':'sha256:base','State':{'Running':False},
                    'Config':{},'HostConfig':{},'Mounts':[
                        {'Destination':'/a','Source':'/first','RW':False},
                        {'Destination':'/b','Source':'/second','RW':True}]}
        reordered = copy.deepcopy(original)
        reordered['Mounts'].reverse()
        self.assertEqual(stable_metadata([original]),stable_metadata([reordered]))
        self.assertEqual(original['Mounts'][0]['Destination'],'/a')

    def test_actual_mutations_remain_detectable(self):
        original = {'Id':'one','Image':'sha256:base','State':{'Running':False},
                    'Config':{},'HostConfig':{},'Mounts':[{'Destination':'/a','Source':'/first','RW':False}]}
        for field,value in [('Image','sha256:changed'),('State',{'Running':True}),
                            ('Mounts',[{'Destination':'/a','Source':'/first','RW':True}])]:
            changed = copy.deepcopy(original)
            changed[field] = value
            self.assertNotEqual(stable_metadata([original]),stable_metadata([changed]))


if __name__=='__main__':
    unittest.main()
