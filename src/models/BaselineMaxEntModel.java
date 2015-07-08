package models;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.json.JSONException;

import types.DataTriplet;
import types.LabelType;
import types.SimpleConfusionMatrix;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.MaxEntTrainer;
import cc.mallet.classify.Trial;
import cc.mallet.types.InstanceList;

public class BaselineMaxEntModel extends CognitiveDiscourseParserBase {

	private Classifier classifier = null;
	
	public BaselineMaxEntModel(String experimentName, String trainingDir, String devDir, String testDir)
			throws JSONException, IOException {
		data = null;
		originalData = loadOriginalData(experimentName, trainingDir, devDir, testDir);
	}

	@Override
	public void trainTest() throws FileNotFoundException {
		// TODO Auto-generated method stub
		ClassifierTrainer<?> trainer = new MaxEntTrainer();
		SimpleConfusionMatrix cm;
		Trial baselineResult;
		for (LabelType labelType : LabelType.values()){
			trainer = new MaxEntTrainer();
			// Evaluate on the combined dimension and mapped to the original label
			originalData.importData(labelType);
			trainer.train(this.originalData.getTrainingSet());
			classifier = trainer.getClassifier();

			System.out.println("====== Baseline classifier performance ("+ labelType +") ======");
			baselineResult = new Trial(classifier, this.originalData.getTestSet());
			cm = new SimpleConfusionMatrix(baselineResult);
			System.out.println(cm.toString());

			baselineResult = new Trial(classifier, this.originalData.getDevSet());
			cm = new SimpleConfusionMatrix(baselineResult);
			System.out.println(cm.toString());
		}
		
	}

	// TODO: Does not support all three label types yet
	@Override
	public String[] classify(String dataFile, LabelType labelType) throws FileNotFoundException {
		// TODO Auto-generated method stub
		InstanceList unseenData = DataTriplet.importData(dataFile, classifier.getInstancePipe());
		int numInstances = unseenData.size();
		String[] predictedLabels = new String[numInstances];
		for (int i = 0; i < numInstances; i++){
			predictedLabels[i] = (String) classifier.classify(unseenData.get(i)).getLabeling().getBestLabel().getEntry();
		}
		return predictedLabels;
	}

	/**
	 * @param args
	 * @throws IOException 
	 * @throws JSONException 
	 */
	public static void main(String[] args) throws JSONException, IOException {
		// TODO Auto-generated method stub
		CognitiveDiscourseParserBase classifier = new BaselineMaxEntModel(
				//"experiment0",
				args[0], 
				"conll15-st-05-19-15-train", "conll15-st-05-19-15-dev", "conll15-st-05-19-15-test");
		classifier.trainTest();
	}



}
