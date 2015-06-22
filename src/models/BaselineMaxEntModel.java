package models;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.json.JSONException;

import types.DataTriplet;
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
	public void trainTest() {
		// TODO Auto-generated method stub
		ClassifierTrainer<?> trainer = new MaxEntTrainer();
		trainer.train(this.originalData.getTrainingSet());
		classifier = trainer.getClassifier();
		Trial baselineResult = new Trial(classifier, this.originalData.getTestSet());
		SimpleConfusionMatrix cm = new SimpleConfusionMatrix(baselineResult);
		System.out.println(cm.toString());
		
	}

	@Override
	public String[] classify(String dataFile) throws FileNotFoundException {
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
				"experiment0",
				//args[0], 
				"conll15-st-05-19-15-train", "conll15-st-05-19-15-dev", "conll15-st-05-19-15-test");
		classifier.trainTest();
	}

}
