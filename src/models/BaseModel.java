package models;

import java.io.FileNotFoundException;

import types.DataTriplet;
import types.SimpleConfusionMatrix;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.Trial;

public abstract class BaseModel {

	public BaseModel() {
		// TODO Auto-generated constructor stub
	}

	public DataTriplet data;
	public Classifier classifier;
	
	public BaseModel(String trainingSet, String devSet, String testSet) {
		data = new DataTriplet(trainingSet, devSet, testSet);
	}
	
	public BaseModel(String featureFileForAllDir) {
		String trainingSet = "conll15-st-05-19-15-train/" + featureFileForAllDir;
		String devSet = "conll15-st-05-19-15-dev/" + featureFileForAllDir;
		String testSet = "conll15-st-05-19-15-test/" + featureFileForAllDir;
		data = new DataTriplet(trainingSet, devSet, testSet);
	}
	
	public abstract ClassifierTrainer<?> getTrainer();
		
	public SimpleConfusionMatrix[] trainTest(boolean importData) throws FileNotFoundException{
		if (importData)data.importData();
		ClassifierTrainer<?> trainer = getTrainer();

		trainer.train(data.getTrainingSet());
		classifier = trainer.getClassifier();
		//PrintWriter pw = new PrintWriter(System.out);
		//classifier.print(pw);
		//classifier.printRank(pw);
		//pw.flush();
		
		Trial devResult = new Trial(classifier, data.getDevSet());
		Trial testResult = new Trial(classifier, data.getTestSet());
		SimpleConfusionMatrix[] results = new SimpleConfusionMatrix[] {new SimpleConfusionMatrix(devResult), new SimpleConfusionMatrix(testResult)};
		System.out.println("Dev set results");
		System.out.println(results[0].toString());
		System.out.println("Test set results");
		System.out.println(results[1].toString());
		return results;
	}
	
	public void trainTest() throws FileNotFoundException{
		trainTest(true);
	}
	
	public Classifier getClassifier() {
		return classifier;
	}

}
