package models;

import java.io.IOException;

import org.json.JSONException;

import types.SimpleConfusionMatrix;

import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.MaxEntTrainer;
import cc.mallet.classify.Trial;

public class BaselineMaxEntModel extends CognitiveDiscourseParserBase {

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
		Classifier classifier = trainer.getClassifier();
		Trial baselineResult = new Trial(classifier, this.originalData.getDevSet());
		SimpleConfusionMatrix cm = new SimpleConfusionMatrix(baselineResult);
		System.out.println(cm.toString());
		
	}

	@Override
	public void classify(String dataDir) {
		// TODO Auto-generated method stub

	}

	/**
	 * @param args
	 * @throws IOException 
	 * @throws JSONException 
	 */
	public static void main(String[] args) throws JSONException, IOException {
		// TODO Auto-generated method stub
		CognitiveDiscourseParserBase classifier = new BaselineMaxEntModel(
				args[0], 
				"conll15-st-05-19-15-train", "conll15-st-05-19-15-dev", "conll15-st-05-19-15-test");
		classifier.trainTest();
	}

}
