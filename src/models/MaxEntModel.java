package models;

import types.Util;
import java.io.FileNotFoundException;

import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.MaxEntTrainer;


public class MaxEntModel extends BaseModel{

	public MaxEntModel(String trainingSet, String devSet, String testSet) {
		super(trainingSet, devSet, testSet);
	}
	public MaxEntModel(String featureFileForAllDir) {
		super(featureFileForAllDir);
	}
	@Override
	public ClassifierTrainer<?> getTrainer() {
		MaxEntTrainer m = new MaxEntTrainer();
		return m;
	}
	

	public static void main(String[] args) throws FileNotFoundException {
		MaxEntModel maxEnt;
		if (args.length > 1) {
			maxEnt = new MaxEntModel(args[0], args[1], args[2]);
		} else {
			maxEnt = new MaxEntModel(args[0]);
		}
		System.out.println("MaxEnt with original data");
		maxEnt.trainTest();
		Util.reweightTrainingDataNway(maxEnt.data.getTrainingSet());
		System.out.println("MaxEnt with reweighted data");
		maxEnt.trainTest();
	}

}
