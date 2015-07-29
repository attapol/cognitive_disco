package types;

import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

public class Util {

	public Util() {
		// TODO Auto-generated constructor stub
	}

	public static void reweightTrainingDataNway(InstanceList data){
		/* Reweighting scheme for n-way classfication
		 * 
		 * Each class will have the same total weight
		 */
		int numClasses = data.getTargetAlphabet().size();
		int numData = data.size();
		int[] trueLabelCount = new int[numClasses];
		for (Instance instance : data) {
			int bestIndex = instance.getLabeling().getBestIndex();
			trueLabelCount[bestIndex]++;
		}
		for (Instance instance : data) {
			int bestIndex = instance.getLabeling().getBestIndex();
			double weight = (numData + 0.0) / trueLabelCount[bestIndex] / numClasses;
			data.setInstanceWeight(instance, weight);
		}
	}
	
}
