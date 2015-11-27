package feature_selection;
/*
 * Information gain of the absence/precence of each feature.

     Modified definition to match with Lin et al., 2009. 
*/


import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Labeling;
import cc.mallet.types.RankedFeatureVector;

public class MInfoGain extends RankedFeatureVector {

	public MInfoGain (InstanceList ilist) {
		super (ilist.getDataAlphabet(), calcInfoGains (ilist));
	}

	public MInfoGain (Alphabet vocab, double[] infogains) {
		super (vocab, infogains);
	}

	private static double[] calcInfoGains (InstanceList ilist) {
		final double log2 = Math.log(2);
		int numInstances = ilist.size();
		int numClasses = ilist.getTargetAlphabet().size();
		int numFeatures = ilist.getDataAlphabet().size();

		double[] infogains = new double[numFeatures];
		double[][] targetFeatureCount = new double[numClasses][numFeatures];
		double[] featureCountSum = new double[numFeatures];
		double[] targetCount = new double[numClasses];
		double targetCountSum = 0;

		int fli; // feature location index
		double count;
		// Populate targetFeatureCount, et al
		for (int i = 0; i < ilist.size(); i++) {
			Instance inst = ilist.get(i);
			Labeling labeling = inst.getLabeling ();
			FeatureVector fv = (FeatureVector) inst.getData ();
			double instanceWeight = ilist.getInstanceWeight(i);
			// The code below relies on labelWeights summing to 1 over all labels!
			double labelWeightSum = 0;
			for (int ll = 0; ll < labeling.numLocations(); ll++) {
				int li = labeling.indexAtLocation (ll);
				double labelWeight = labeling.valueAtLocation (ll);
				labelWeightSum += labelWeight;
				if (labelWeight == 0) continue;
				count = labelWeight * instanceWeight;
				for (int fl = 0; fl < fv.numLocations(); fl++) {
					fli = fv.indexAtLocation(fl);
					// xxx Is this right?  What should we do about negative values?
					// Whatever is decided here should also go in DecisionTree.split()
					if (fv.valueAtLocation(fl) > 0) {
						targetFeatureCount[li][fli] += count;
						featureCountSum[fli] += count;
					}
				}
				targetCount[li] += count;
				targetCountSum += count;
			}
			assert (Math.abs (labelWeightSum - 1.0) < 0.0001);
		}
		if (targetCountSum == 0) {
			return infogains;
		}
		assert (targetCountSum > 0) : targetCountSum;

        double[] baseEntropy = new double[numClasses];
		for (int li = 0; li < numClasses; li++) {
			double p = targetCount[li] / targetCountSum;
			assert (p <= 1.0) : p;
			if (p != 0) {
				baseEntropy[li] -= p * Math.log(p);
                baseEntropy[li] -= (1 - p) * Math.log(1 - p);
            }else {
                baseEntropy[li] = 0;
            }
		}


        for (int fi = 0; fi < numFeatures; fi++) {
            double bestMI = Double.MIN_VALUE;
		    for (int li = 0; li < numClasses; li++) {
                double featurePresentEntropy = 0;
                double featureAbsentEntropy = 0;
			    double norm = featureCountSum[fi];
                if (norm > 0) {
                    double p = targetFeatureCount[li][fi] / norm;
                    assert (p <= 1.00000001) : p; 
                    if (p != 0) {
                        featurePresentEntropy -= p * Math.log(p);
                        featurePresentEntropy -= (1 - p) * Math.log(1 - p);
                    }
                }
			    norm = targetCountSum-featureCountSum[fi];
                if (norm > 0) {
                    double p = (targetCount[li] - targetFeatureCount[li][fi]) / norm;
                    assert (p <= 1.00000001) : p; 
                    if (p != 0) {
                        featureAbsentEntropy -= p * Math.log(p);
                        featureAbsentEntropy -= (1 - p) * Math.log(1 - p);
                    }
                }
                double conditionalEntropy = (featureCountSum[fi]/targetCountSum) * featurePresentEntropy
											 + ((targetCountSum-featureCountSum[fi])/targetCountSum) * featureAbsentEntropy;
                double MI = baseEntropy[li] - conditionalEntropy;
                if (MI > bestMI) infogains[fi] = MI;
            }
        }
        return infogains;
    }

	
}
