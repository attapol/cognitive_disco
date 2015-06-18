package models;

import java.io.IOException;
import java.util.ArrayList;

import org.json.JSONException;

import types.SimpleConfusionMatrix;

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
	
	abstract public ClassifierTrainer<?> getNewTrainer();

	public void trainTest(){
		int numDimensions = this.data.length;
		Trial[] trials = new Trial[numDimensions];
		for (int i = 0; i < numDimensions ; i++) {
			ClassifierTrainer<?> trainer = getNewTrainer();
			trainer.train(this.data[i].getTrainingSet());
			Classifier classifier = trainer.getClassifier();
			trials[i] = new Trial(classifier, this.data[i].getDevSet());
		}
		
		ClassifierTrainer<?> trainer = getNewTrainer();
		trainer.train(this.originalData.getTrainingSet());
		Classifier classifier = trainer.getClassifier();
		Trial baselineResult = new Trial(classifier, this.originalData.getDevSet());
		
		for (Trial trial : trials){
			ConfusionMatrix cm = new ConfusionMatrix(trial);
			System.out.println(cm.toString());
		}
		int numInstances = trials[0].size();
		
		String[] trueLabels = new String[numInstances];
		String[] predictedLabels = new String[numInstances];
		for (int i = 0; i < numInstances; i++){
			ArrayList<String> dimensions = new ArrayList<String>();
			for (int d = 0; d < numDimensions; d++){
				dimensions.add((String) trials[d].get(i).getLabeling().getBestLabel().getEntry());
			}
			predictedLabels[i] = dm.getTopLevelLabel(dimensions);
			
			dimensions = new ArrayList<String>();
			for (int d = 0; d < numDimensions; d++){
				dimensions.add((String) trials[d].get(i).getInstance().getLabeling().getBestLabel().getEntry());
			}
			trueLabels[i] = dm.getTopLevelLabel(dimensions);
		}
		SimpleConfusionMatrix cm = new SimpleConfusionMatrix(trueLabels, predictedLabels);
		System.out.println(cm.toString());
		
		cm = new SimpleConfusionMatrix(baselineResult);
		System.out.println(cm.toString());
		
	}

}
