package models;

import java.io.FileNotFoundException;

import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.MaxEntTrainer;
import cc.mallet.classify.Trial;
import cc.mallet.classify.evaluate.ConfusionMatrix;
import types.DataTriplet;

public class MaxEntModel {

	DataTriplet data;
	public MaxEntModel(String trainingSet, String devSet, String testSet) {
		// TODO Auto-generated constructor stub
		data = new DataTriplet(trainingSet, devSet, testSet);
	}
	
	public MaxEntModel(String featureFileForAllDir) {
		String trainingSet = "conll15-st-05-19-15-train/" + featureFileForAllDir;
		String devSet = "conll15-st-05-19-15-dev/" + featureFileForAllDir;
		String testSet = "conll15-st-05-19-15-test/" + featureFileForAllDir;
		data = new DataTriplet(trainingSet, devSet, testSet);
	}
	
	public void trainTest() throws FileNotFoundException{
		data.importData();
		ClassifierTrainer<?> trainer = new MaxEntTrainer();
		trainer.train(data.getTrainingSet());
		Classifier classifier = trainer.getClassifier();
		
		Trial devResult = new Trial(classifier, data.getDevSet());
		Trial testResult = new Trial(classifier, data.getTestSet());
		System.out.println("Dev set results");
		System.out.println(new ConfusionMatrix(devResult).toString());
		System.out.println("Test set results");
		System.out.println(new ConfusionMatrix(testResult).toString());
		
	}

	public static void main(String[] args) throws FileNotFoundException {
		//MaxEntModel maxEnt = new MaxEntModel(args[0], args[1], args[2]);
		MaxEntModel maxEnt = new MaxEntModel(args[0]);
		maxEnt.trainTest();
	}

}
