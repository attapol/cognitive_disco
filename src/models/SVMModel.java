package models;

import java.io.FileNotFoundException;

import ca.uwo.csd.ai.nlp.kernel.LinearKernel;
import ca.uwo.csd.ai.nlp.mallet.libsvm.SVMClassifierTrainer;
import cc.mallet.classify.ClassifierTrainer;

public class SVMModel extends BaseModel {

	public SVMModel(String trainingSet, String devSet, String testSet) {
		super(trainingSet, devSet, testSet);
	}

	public SVMModel(String featureFileForAllDir) {
		super(featureFileForAllDir);
	}

	@Override
	public ClassifierTrainer<?> getTrainer() {
		SVMClassifierTrainer trainer = new SVMClassifierTrainer(new LinearKernel());
		return trainer;
	}

	public static void main(String[] args) throws FileNotFoundException{
		SVMModel model;
		if (args.length > 1) {
			model = new SVMModel(args[0], args[1], args[2]);
		} else {
			model = new SVMModel(args[0]);
		}
		model.trainTest(true);

	}

}
