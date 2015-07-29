package models;

import java.io.FileNotFoundException;

import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.NaiveBayesTrainer;

public class NaiveBayesModel extends BaseModel {

	public NaiveBayesModel() {
		// TODO Auto-generated constructor stub
	}

	public NaiveBayesModel(String trainingSet, String devSet, String testSet) {
		super(trainingSet, devSet, testSet);
	}

	public NaiveBayesModel(String featureFileForAllDir) {
		super(featureFileForAllDir);
	}

	@Override
	public ClassifierTrainer<?> getTrainer() {
		return new NaiveBayesTrainer();
	}

	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub
		NaiveBayesModel m;
		if (args.length > 1) {
			m = new NaiveBayesModel(args[0], args[1], args[2]);
		} else {
			m = new NaiveBayesModel(args[0]);
		}
		m.trainTest();
	}

}
