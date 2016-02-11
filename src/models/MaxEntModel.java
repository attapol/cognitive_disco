package models;

import types.Util;

import java.io.FileNotFoundException;

import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.MaxEnt;
import cc.mallet.classify.MaxEntTrainer;


public class MaxEntModel extends BaseModel{

	public MaxEntModel(String trainingSet, String devSet, String testSet) {
		super(trainingSet, devSet, testSet);
	}

    public MaxEntModel(String partialDir, String featureFileForAllDir) {
        super(partialDir, featureFileForAllDir);
    }

	public MaxEntModel(String featureFileForAllDir) {
		super(featureFileForAllDir);
	}
	@Override
	public ClassifierTrainer<?> getTrainer() {
		MaxEntTrainer m = new MaxEntTrainer();
		return m;
	}
	
	
	public double[] getParameters() {
		MaxEnt maxent = (MaxEnt) classifier;
		return maxent.getParameters();
	}

	public static void main(String[] args) throws FileNotFoundException {
		MaxEntModel m;
		if (args.length == 3) {
			m = new MaxEntModel(args[0], args[1], args[2]);
		}else if (args.length == 2) {
			m = new MaxEntModel(args[0], args[1]);
        }else if (args.length == 1) {
			m = new MaxEntModel(args[0]);
		}else {
            System.err.println("Wrong number of arguments");
            return;
        }
		System.out.println("MaxEnt with original data");
		m.trainTest(true);
		/*
		Util.reweightTrainingDataNway(maxEnt.data.getTrainingSet());
		System.out.println("MaxEnt with reweighted data");
		maxEnt.trainTest(false);
		*/
	}

}
