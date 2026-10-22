import copy
import unittest
from studies.ml_three_track_v1.track_a import HOUR, enrich


class TemporalTests(unittest.TestCase):
    def row(self,dep,available,label=1,airport='A'):
        return {'origin':airport,'destination':airport,'prediction':dep,'available':available,'label':label}

    def test_exact_bounds_future_inflight_and_duplicates(self):
        t=10*HOUR
        sources=[self.row(t-HOUR,t-1),self.row(t-HOUR,t-1),self.row(t-HOUR-1,t-1),
                 self.row(t-1,t),self.row(t-1,t+1),self.row(t,t-1),self.row(t-1,None),
                 self.row(t-3*HOUR,t-1,0),self.row(t-3*HOUR-1,t-1)]
        target=self.row(t,t+HOUR)
        values=enrich([target,target],sources)
        self.assertEqual(len(values),2)
        for role in ('origin','destination'):
            self.assertEqual(values[0][1][role+'_1h'],{'count':2,'rate':1.,'missing':0})
            self.assertEqual(values[0][1][role+'_3h'],{'count':4,'rate':.75,'missing':0})
        changed=copy.deepcopy(sources)
        for s in changed:
            if s['prediction']>=t or s['available'] is None or s['available']>=t:s['label']=1-s['label']
        self.assertEqual(values,enrich([target,target],changed))

    def test_null_time_empty_and_null_airport(self):
        values=enrich([self.row(None,None),self.row(10,20,airport=None)],[self.row(0,1,airport=None)])
        self.assertEqual(values[0][1]['origin_1h'],{'count':0,'rate':None,'missing':1})
        self.assertEqual(values[1][1]['origin_1h']['count'],1)

    def test_early_availability_never_admits_future_departure(self):
        target=self.row(100,200)
        self.assertEqual(enrich([target],[self.row(101,0)])[0][1]['origin_1h']['count'],0)

    def test_order_invariance_and_identity_preservation(self):
        targets=[self.row(100,200),self.row(100,200)]
        sources=[self.row(0,1),self.row(50,51,0)]
        a=enrich(targets,sources);b=enrich(targets,list(reversed(sources)))
        self.assertEqual(a,b)
        self.assertTrue(all(x is y[0] for x,y in zip(targets,a)))


if __name__=='__main__':unittest.main()
