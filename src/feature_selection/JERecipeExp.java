package feature_selection;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSelection;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.InfoGain;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.RankedFeatureVector;
import types.DataTriplet;
import types.LabelType;
import models.MaxEntModel;

public class JERecipeExp {

	public JERecipeExp() {
		
	}

	public void run() throws FileNotFoundException, UnsupportedEncodingException {
		InfoGainSelectFeatures[] igss = new InfoGainSelectFeatures[] {
				new InfoGainSelectFeatures("production_rules.original_label.features", 100),
				new InfoGainSelectFeatures("word_pairs.original_label.features", 500),
				new InfoGainSelectFeatures("dependency_rules.original_label.features", 100),
				new InfoGainSelectFeatures("brown_word_pairs.original_label.features", 600),
		};
		HashMap<String, Integer> featureToIndex = new HashMap<String, Integer>();
		String fileName = "je_recipe.sfeatures";
		createSparseMatrix(igss, DataTriplet.DataSplitType.TRAINING, fileName, featureToIndex, false);
		createSparseMatrix(igss, DataTriplet.DataSplitType.DEV, fileName, featureToIndex, false);
		createSparseMatrix(igss, DataTriplet.DataSplitType.TEST, fileName, featureToIndex, false);
	
		fileName = "je_recipe.features";
		createSparseMatrix(igss, DataTriplet.DataSplitType.TRAINING, fileName, featureToIndex, true);
		createSparseMatrix(igss, DataTriplet.DataSplitType.DEV, fileName, featureToIndex, true);
		createSparseMatrix(igss, DataTriplet.DataSplitType.TEST, fileName, featureToIndex, true);
		
	}

	public void createSparseMatrix(InfoGainSelectFeatures[] igss, DataTriplet.DataSplitType dType,
			String fileName, HashMap<String, Integer> featureToIndex, boolean printLabel) throws FileNotFoundException, UnsupportedEncodingException {
		String dirName = igss[0].model.data.getDataFileName(dType).split("/")[0];
		String fullPath = dirName + "/" + fileName;
		PrintWriter writer = new PrintWriter(new File(fullPath));	
		int numInstances = igss[0].model.data.getData(dType).size();
		for (int i = 0; i < numInstances; i++) {
			String currentInstanceName = "";
			for (int j = 0; j < igss.length; j++) {
				InfoGainSelectFeatures igs = igss[j];
				BitSet bs = igs.featureSelection.getBitSet();
				Instance inst = igs.model.data.getData(dType).get(i);
				String name = (String)inst.getName();
				if (j == 0) {
					writer.write(name);
					currentInstanceName = name;
					if (printLabel){ 
						String label = inst.getLabeling().toString();
						writer.write("\t" + label +"\t");
					}
				} else {
					assert(name.equals(currentInstanceName));
				}
				FeatureVector fv = (FeatureVector)inst.getData();
				Alphabet alphabet = fv.getAlphabet();
				int[] features = fv.getIndices();
				int numFeaturesAdded = 0;
				for (int fi : features) {
					boolean selected = bs.get(fi);
					if (selected) {
						String featureName = alphabet.lookupObject(fi).toString();
						if (!featureToIndex.containsKey(featureName)) featureToIndex.put(featureName, featureToIndex.size());
						int newIndex = featureToIndex.get(featureName);
						writer.write(" "+newIndex);
						numFeaturesAdded++;
					}
				}
				if (numFeaturesAdded == 0) writer.write(" "+bs.cardinality());
			}
			writer.write("\n");
		}
		writer.close();

	}
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		JERecipeExp experiment = new JERecipeExp();
		experiment.run();
	}

}
