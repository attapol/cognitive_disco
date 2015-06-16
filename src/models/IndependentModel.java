package models;

import java.io.IOException;
import org.json.JSONException;

import cc.mallet.classify.evaluate.ConfusionMatrix;

import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.Trial;


public abstract class IndependentModel extends CognitiveDiscourseParserBase{

	public IndependentModel(String experimentName, String mappingJson,
			String trainingDir, String devDir, String testDir)
			throws JSONException, IOException {
		super(experimentName, mappingJson, trainingDir, devDir, testDir);
	}
	
	abstract public ClassifierTrainer getNewTrainer();

	public void trainTest(){
		for (int i = 0; i < this.data.length; i++){
			ClassifierTrainer trainer = getNewTrainer();
			trainer.train(this.data[i].getTrainingSet());
			Classifier classifier = trainer.getClassifier();
			Trial trial = new Trial(classifier, this.data[i].getDevSet());
			ConfusionMatrix cm = new ConfusionMatrix(trial);
			System.out.println(cm.toString());
		}
	}

}
