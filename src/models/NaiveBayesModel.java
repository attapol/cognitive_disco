package models;

import types.Util;
import java.io.FileNotFoundException;

import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.NaiveBayesTrainer;

public class NaiveBayesModel extends BaseModel {

	public NaiveBayesModel() {
		// TODO Auto-generated constructor stub
	}
    public NaiveBayesModel(String partialDir, String featureFileForAllDir) {
        super(partialDir, featureFileForAllDir);
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
		if (args.length == 3) {
			m = new NaiveBayesModel(args[0], args[1], args[2]);
		} else if (args.length == 2) {
			m = new NaiveBayesModel(args[0], args[1]);
        } else if (args.length == 1) {
			m = new NaiveBayesModel(args[0]);
		} else {
            System.err.println("Wrong number of arguments");
            return;
        }
		System.out.println("NaiveBayes with original data");
		m.trainTest(true);
		//Util.reweightTrainingDataNway(m.data.getTrainingSet());
		//System.out.println("NaiveBayes with reweighted data");
		//m.trainTest(false);
	}

}
