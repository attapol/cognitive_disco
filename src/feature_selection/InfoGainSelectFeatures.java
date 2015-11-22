package feature_selection;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.BitSet;
import java.util.HashMap;

import models.MaxEntModel;
import types.LabelType;
import cc.mallet.classify.MaxEnt;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSelection;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.InfoGain;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.RankedFeatureVector;

public class InfoGainSelectFeatures {

	public MaxEntModel model; 
	private String experimentName;
	private int numFeatures;
	public FeatureSelection featureSelection;

	public InfoGainSelectFeatures(String experimentName, int numFeatures) throws FileNotFoundException {
		// TODO Auto-generated constructor stub
		model = new MaxEntModel(experimentName);
		this.experimentName = experimentName;
		this.numFeatures = numFeatures;

		model.data.importData(LabelType.SCHEME_B);
		RankedFeatureVector infoGain = new InfoGain(model.data.getTrainingSet());
		featureSelection = new FeatureSelection(infoGain, numFeatures);
		model.data.setFeatureSelection(featureSelection);
	//	model.trainTest(false);
	}
	
	public void writeFeatures() throws FileNotFoundException {
		HashMap<Integer, Integer> oldToNewMapping = new HashMap<Integer, Integer>();
		String newFileName = getFileName(model.data.getTrainingFileName(), experimentName, numFeatures);
		String modelFileName = newFileName+".model";
		createSparseMatrix(model.data.getTrainingSet(), featureSelection, newFileName, oldToNewMapping);

		newFileName = getFileName(model.data.getDevFileName(), experimentName, numFeatures);
		createSparseMatrix(model.data.getDevSet(), featureSelection, newFileName, oldToNewMapping);

		newFileName = getFileName(model.data.getTestFileName(), experimentName, numFeatures);
		createSparseMatrix(model.data.getTestSet(), featureSelection, newFileName, oldToNewMapping);
		assert(oldToNewMapping.size() == numFeatures);
	
		writeModelParameters(modelFileName, model, oldToNewMapping);
	}
	
	public String getFileName(String dataFileName, String experimentName, int numFeatures) {
		String dirName = dataFileName.split("/")[0];
		return dirName+"/"+experimentName+"_"+numFeatures+"f.sparse";
	}
	
	public static final String[] VALID_SENSES = new String[] {
		"Comparison.Concession",
		"Comparison.Contrast",
		"Contingency.Cause",
		"Contingency.Pragmatic cause",
		"Expansion.Alternative",
		"Expansion.Conjunction",
		"Expansion.Instantiation",
		"Expansion.List",
		"Expansion.Restatement",
		"Temporal.Asynchronous",
		"Temporal.Synchrony"
	};

	
	public void writeModelParameters(String modelFileName, MaxEntModel model, HashMap<Integer, Integer> oldToNewMapping) throws FileNotFoundException {
		MaxEnt maxent = (MaxEnt) model.getClassifier();
		LabelAlphabet la = maxent.getLabelAlphabet();
		HashMap<String, Integer> newla = new HashMap<String, Integer>();
		for (int i = 0; i < VALID_SENSES.length; i++) newla.put(VALID_SENSES[i], i);
		
		double[] parameters = maxent.getParameters();
		int numLabels = maxent.getNumParameters() / (maxent.getDefaultFeatureIndex() + 1);
		int numFeatures = maxent.getDefaultFeatureIndex() + 1;
		int numColumns = oldToNewMapping.size() + 1;
		double[][] W = new double[VALID_SENSES.length][numColumns];
		for (int li = 0; li < numLabels; li++) {

			String label = la.lookupLabel(li).toString();
			if (newla.containsKey(label)) {
				int newLi = newla.get(label);
				for (int oldIndex : oldToNewMapping.keySet()){
					int newIndex = oldToNewMapping.get(oldIndex);
					int paramIndex = li * numFeatures + oldIndex;
					W[newLi][newIndex] = parameters[paramIndex];
				}
				W[newLi][numColumns-1] = parameters[li * numFeatures + numFeatures-1];
			} else {
				System.err.println(label);
			}
		}

		PrintWriter writer = new PrintWriter(new File(modelFileName));	
		for (int i = 0; i < W.length; i++){
			for (int j = 0; j < W[i].length; j++){
				writer.write(Double.toString(W[i][j]));
				if (j != W[i].length -1) writer.write(", ");
			}
			writer.write("\n");
		}
		writer.close();
	}
	

	
	public void createSparseMatrix(InstanceList data, FeatureSelection fs, 
			String fileName, HashMap<Integer, Integer> oldToNewMapping) throws FileNotFoundException{
		BitSet bs = fs.getBitSet(); 
		PrintWriter writer = new PrintWriter(new File(fileName));	
		for (Instance inst : data) {
			String name = (String)inst.getName();
			writer.write(name);
			FeatureVector fv = (FeatureVector)inst.getData();
			int[] features = fv.getIndices();
			for (int fi : features) {
				boolean selected = bs.get(fi);
				if (selected) {
					if (!oldToNewMapping.containsKey(fi)) oldToNewMapping.put(fi, oldToNewMapping.size());
					int newIndex = oldToNewMapping.get(fi);
					writer.write(" "+newIndex);
				}
			}
			writer.write("\n");
		}
		writer.close();
	}

	public static void main(String[] args) throws NumberFormatException, FileNotFoundException {
		// TODO Auto-generated method stub
		InfoGainSelectFeatures ig = new InfoGainSelectFeatures(args[0], Integer.parseInt(args[1]));
		ig.writeFeatures();

	}

}
