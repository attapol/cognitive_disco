package types;

import java.util.ArrayList;

import cc.mallet.classify.Classifier;
import cc.mallet.classify.Classification;
import cc.mallet.classify.Trial;
import cc.mallet.types.Instance;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.Alphabet;
import cc.mallet.types.LabelVector;
import cc.mallet.types.Labeling;
import cc.mallet.types.MatrixOps;
import org.json.*;

/**
 * The implementation of ConfusionMatrix is borrowed from the one in Mallet
 * This version handles multiple Trials and combines results (usually from CV)
 * @author tet
 *
 */
public class SimpleConfusionMatrix{
	int[][] values;
	int numClasses;
	Alphabet labelAlphabet;
	Classifier classifier = null;
	
	public SimpleConfusionMatrix(Trial[] trials)
	{
		for (int i = 0; i < trials.length; i++){
			Trial trial = trials[i];
			ArrayList<Classification> classifications = trial;
			if (i == 0){
				classifier = trial.getClassifier();
				labelAlphabet = classifier.getLabelAlphabet();
				Labeling tempLabeling = classifications.get(0).getLabeling();
				numClasses = tempLabeling.getLabelAlphabet().size();
				values = new int[numClasses][numClasses];
			}
			
			for(int j=0; j < classifications.size(); j++)
			{
				LabelVector lv = classifications.get(j).getLabelVector();
				Instance inst = classifications.get(j).getInstance();
				int bestIndex = lv.getBestIndex();
				int correctIndex = inst.getLabeling().getBestIndex();
				assert(correctIndex != -1);
				values[correctIndex][bestIndex]++;
			}	
		}
	}
	
	public SimpleConfusionMatrix(Alphabet dict, int[][] matrixValues)
	{
		values = matrixValues;
		numClasses = matrixValues.length;
		labelAlphabet = dict;
	}
	
	public SimpleConfusionMatrix(String[] trueLabels, String[] predictedLabels)
	{
		assert (trueLabels.length == predictedLabels.length);
		int numInstances = trueLabels.length;
		
		labelAlphabet = new Alphabet();
		for (String label : trueLabels) labelAlphabet.lookupIndex(label); 
		labelAlphabet.stopGrowth();
		numClasses = labelAlphabet.size();
		values = new int[numClasses][numClasses];
		for (int i = 0; i < numInstances; i++){
			int trueIndex = labelAlphabet.lookupIndex(trueLabels[i]);
			int predictedIndex = labelAlphabet.lookupIndex(predictedLabels[i]);
			values[trueIndex][predictedIndex] ++;
		}
		
	}
	
	public SimpleConfusionMatrix(Trial trial)
	{
		this(new Trial[]{trial});
	}

	public Alphabet getLabelAlphabet(){
		return labelAlphabet;
	}
	/* Merge two labels into one 
	 *
	 * Merge label1 and label2 into finalLabel
	 * Recalculate the values in the matrix and recreate the LabelAlphabet
	 */
	public void mergeLabels(String label1, String label2, String finalLabel)
	{
		int labelIndex1 = labelAlphabet.lookupIndex(label1, false);
		int labelIndex2 = labelAlphabet.lookupIndex(label2, false);
		if (labelIndex1 == -1 || labelIndex2 == -1) return;
		
		LabelAlphabet newAlphabet = new LabelAlphabet();
		for (int i = 0; i < numClasses; i++){
			Object label = labelAlphabet.lookupObject(i);
			if (i == labelIndex1 || i == labelIndex2){
				newAlphabet.lookupIndex(finalLabel, true);
			} else{
				newAlphabet.lookupIndex(label, true);
			}
		}
		int[][] newMatrix = new int[numClasses-1][numClasses-1];
		for (int i = 0; i < numClasses; i++){
			for (int j=0; j < numClasses; j++){
				int newi = (i == labelIndex1 || i == labelIndex2) ? 
						newAlphabet.lookupIndex(finalLabel): newAlphabet.lookupIndex(labelAlphabet.lookupObject(i), false);
				int newj = (j == labelIndex1 || j == labelIndex2) ? 
						newAlphabet.lookupIndex(finalLabel): newAlphabet.lookupIndex(labelAlphabet.lookupObject(j), false);
				newMatrix[newi][newj] += values[i][j];
			}
		}
		values = newMatrix;
		labelAlphabet = newAlphabet;
		numClasses--;
	}

	public int getNumOccurrences(int classIndex)
	{
		int total = 0;
		for (int trueClassIndex=0; trueClassIndex < this.numClasses; trueClassIndex++) {
			total += values[trueClassIndex][classIndex];
		}
		return total;
	}

	public double getAccuracy ()
	{
		int correct = 0;
		int total = 0;
		for (int i = 0; i < this.numClasses; i++){
			for (int j = 0; j < this.numClasses; j++){
				if (i == j) correct += this.values[i][j];
				total += this.values[i][j];
			}
		}
		double accuracy = (double) correct / total;
		return round(accuracy, 4);
	}
	
	public double getPrecision (int predictedClassIndex)
	{
		int total = 0;
		for (int trueClassIndex=0; trueClassIndex < this.numClasses; trueClassIndex++) {
			total += values[trueClassIndex][predictedClassIndex];
		}
		if (total == 0){
			return 0.0;
		} else {
			double precision = (double) (values[predictedClassIndex][predictedClassIndex]) / total;
			return round(precision, 4);
		}
	}
	
	public double getRecall (int trueClassIndex)
	{
		int total = 0;
		for (int predictedClassIndex=0; predictedClassIndex < this.numClasses; predictedClassIndex++) {
			total += values[trueClassIndex][predictedClassIndex];
		}
		if (total == 0){
			return 0.0;
		}else{
			double recall = (double) (values[trueClassIndex][trueClassIndex]) / total;
			return round(recall, 4);
		}
	}
	
	public double getF1 (int classIndex){
		double precision = getPrecision(classIndex);
		double recall = getRecall(classIndex);
		double f1 = 0.0;
		if (precision + recall != 0)
			f1 = 2 * (precision * recall) / (precision + recall);
		return round(f1, 4);
	}
	
	public double getMacroAverageF1(){
		double[] F1s = getAllF1();
		double sumF1 = 0.0;
		for (double F1 : F1s) sumF1 += F1;
		return sumF1 / F1s.length;
	}

	public double[] getAllF1(){
		double[] results = new double[numClasses];
		for (int i = 0; i < results.length; i++){
			results[i] = getF1(i);
		}
		return results;
	}
	
	public int getNumClasses(){
		return numClasses;
	}
	
	public double round(double val, int decimalPoints){
		double factor = Math.pow(10, decimalPoints);
		return Math.round(val * factor)/factor;
	}

	public String getPerformanceReport(){
		StringBuilder sb = new StringBuilder ();
		if (classifier != null) {
			String newString = classifier.getClass().getName() + "\n";
			sb.append(newString);
		}
		double accuracy = getAccuracy();
		double macroF1 = 0.0;
		double microF1 = 0.0;
		int datasetSize = 0;
		for (int i = 0; i < labelAlphabet.size(); i++){
			sb.append((String) labelAlphabet.lookupObject(i));
			//sb.append((String) labelAlphabet.lookupLabel(i).getEntry());
			sb.append("\t");
			double precision = getPrecision(i); double recall = getRecall(i);
			double f1 = 0.0;
			if (precision + recall != 0)
				f1 = 2*(precision * recall)/(precision + recall);
			macroF1 += f1;
			int numClassOccurrences = getNumOccurrences(i);
			microF1 += f1 * numClassOccurrences;
			datasetSize += numClassOccurrences;
			sb.append("precision " + round(getPrecision(i),4) + "\t recall " + round(getRecall(i),4) + "\t F1 " + round(f1,4) + "\n");
		}
		macroF1 = macroF1 / labelAlphabet.size();
		microF1 = microF1 / datasetSize;
		sb.append("\naccuracy " + round(accuracy,4) + "\nmacro-average F1 " + round(macroF1,4) + "\nmicro-average F1 "+ round(microF1, 4) + "\n");
		return sb.toString();
	}
	
	/*
	 * Returns the json string with performance report
	 * For example, 
	 * "{ 'class1' : { 'precision' : 0.1234, 'recall' : 0.4321, 'f1' : 0.3421 } 
	 * 	 'class2' : { 'precision' : 0.1234, 'recall' : 0.4321, 'f1' : 0.3421 } 
	 * 	 'overall' : { 'accuracy' : 0.1234, 'f1' : 0.4321}
	 * }"
	 */
	public String getPerformanceReportJson() throws JSONException {
		JSONObject dict = new JSONObject();
		double accuracy = getAccuracy();
		double macroF1 = 0.0;
		double microF1 = 0.0;
		int datasetSize = 0;
		for (int i = 0; i < labelAlphabet.size(); i++){
			String labelName = (String) labelAlphabet.lookupObject(i);
			double precision = getPrecision(i); double recall = getRecall(i);
			double f1 = 0.0;
			if (precision + recall != 0)
				f1 = 2*(precision * recall)/(precision + recall);
			
			JSONObject metricDict = new JSONObject();
			metricDict.put("precision", round(precision,4));
			metricDict.put("recall", round(recall, 4));
			metricDict.put("f1", round(f1, 4));
			macroF1 += f1;
			int numClassOccurrences = getNumOccurrences(i);
			microF1 += f1 * numClassOccurrences;
			datasetSize += numClassOccurrences;
			dict.put(labelName, metricDict);
		}
		macroF1 = macroF1 / labelAlphabet.size();
		microF1 = microF1 / datasetSize;
		JSONObject metricDict = new JSONObject();
		metricDict.put("accuracy", round(accuracy,4));
		metricDict.put("macro-f1", round(macroF1,4));
		metricDict.put("micro-f1", round(microF1,4));
		dict.put("overall", metricDict);
		return dict.toString();
	}
	
	public String getMatrixString(){
		StringBuffer sb = new StringBuffer();
		int maxLabelNameLength = 0;
		for (int i = 0; i < numClasses; i++) {
			int len = labelAlphabet.lookupObject(i).toString().length();
			if (maxLabelNameLength < len)
				maxLabelNameLength = len;
		}
		
		//sb.append ("============================================================\n");
		double[] distribution = new double[values.length];
		for (int i = 0; i < distribution.length; i++)
			distribution[i] = MatrixOps.sum(values[i]);
		double baselineAccuracy = MatrixOps.max(distribution) / MatrixOps.sum(distribution);
		sb.append ("Confusion Matrix, row=true, column=predicted  accuracy="+getAccuracy()+" most-frequent-tag baseline="+baselineAccuracy+"\n");
		
		for (int i = 0; i < maxLabelNameLength-5+4; i++) sb.append (' ');
		sb.append ("label");
		for (int c2 = 0; c2 < Math.min(10,numClasses); c2++)	sb.append ("   "+c2);
		for (int c2 = 10; c2 < numClasses; c2++)	sb.append ("  "+c2);
		sb.append ("  |total\n");
		for (int c = 0; c < numClasses; c++) {
			appendJustifiedInt (sb, c, false);
			//String labelName = labelAlphabet.lookupLabel(c).toString();
			String labelName = labelAlphabet.lookupObject(c).toString();
			for (int i = 0; i < maxLabelNameLength-labelName.length(); i++) sb.append (' ');
			sb.append (" "+labelName+" ");
			for (int c2 = 0; c2 < numClasses; c2++) {
				appendJustifiedInt (sb,  values[c][c2], true);
				sb.append (' ');
			}
			sb.append (" |"+ MatrixOps.sum(values[c]));
			sb.append ('\n');
		}
		return sb.toString();
	}
	
	public String toString () {
		StringBuffer sb = new StringBuffer ();
		sb.append(getMatrixString());
		sb.append(getPerformanceReport());
		return sb.toString();
	}
	
	private void appendJustifiedInt (StringBuffer sb, int i, boolean zeroDot) {
		if (i < 100) sb.append (' ');
		if (i < 10) sb.append (' ');
		if (i == 0 && zeroDot)
			sb.append (".");
		else
			sb.append (""+i);
	}
	
	public static void main(String[] args){
		/*
		 * Testing the merging function
		 */
		LabelAlphabet alphabet = new LabelAlphabet();
		alphabet.lookupIndex("Class A");
		alphabet.lookupIndex("Class B");
		alphabet.lookupIndex("Class C");
		int[][] values = { {3, 4, 5},
				{7, 8, 9},
				{10, 11, 12}
		};
		SimpleConfusionMatrix cm = new SimpleConfusionMatrix(alphabet, values);		
		System.out.println("Test simple confusion matrix");
		System.out.println(cm.toString());
		
		System.out.println("Merging class B and class C");
		cm.mergeLabels("Class C", "Class B", "Class Special");
		System.out.println("Merging...");
		System.out.println(cm.toString());
		
		String[] trueLabels = new String[] { "A", "B", "B", "C" };
		String[] predictedLabels = new String[] { "A", "B", "C", "C" };
		
		cm = new SimpleConfusionMatrix(trueLabels, predictedLabels);
		System.out.println(cm.toString());
	}

}
