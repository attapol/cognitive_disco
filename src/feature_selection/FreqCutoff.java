package feature_selection;
/*
 * Select Features based on their frequency
 * We want to screen out features that occur too rarely.
 */

import java.util.BitSet;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSelection;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

public class FreqCutoff{

	private Alphabet alphabet;
	private double[] frequencies;
	
	public FreqCutoff (InstanceList ilist) {
		alphabet = ilist.getDataAlphabet();
		countFrequency(ilist);
	}

	
	public FeatureSelection getFeatureSelection(double cutoff) {
		BitSet selectedFeatures = new BitSet();
		for (int i = 0; i < frequencies.length; i++) 
			selectedFeatures.set(i, frequencies[i] >= cutoff);
		return new FeatureSelection(alphabet, selectedFeatures);
	}
	
	
	public void countFrequency(InstanceList ilist) {
		frequencies = new double[alphabet.size()];
		for (Instance instance: ilist){
			FeatureVector data = (FeatureVector) instance.getData();
			for (int index: data.getIndices()) 
				frequencies[index]++;
		}
		
	}
	


}
