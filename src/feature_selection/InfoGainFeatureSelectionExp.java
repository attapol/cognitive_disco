package feature_selection;

import java.io.IOException;
import java.util.HashMap;

import org.json.JSONException;

import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.types.FeatureSelection;
import cc.mallet.types.InfoGain;
import cc.mallet.types.RankedFeatureVector;
import types.DataTriplet;
import types.LabelType;
import types.ResultWriter;
import types.SimpleConfusionMatrix;
import models.MaxEntModel;

public class InfoGainFeatureSelectionExp {

	public InfoGainFeatureSelectionExp(String experimentName, String resultName) throws JSONException, IOException {
		//BaselineMaxEntModel model = new BaselineMaxEntModel(experimentName, trainingDir, devDir, testDir);
		MaxEntModel model = new MaxEntModel(experimentName);
		model.data.importData(LabelType.SCHEME_B);
		RankedFeatureVector infoGain = new InfoGain(model.data.getTrainingSet());
		ResultWriter rw = new ResultWriter(resultName);
		//int[] numFeatureList = new int[]{1000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 40000, 50000, 100000}; 
		int[] numFeatureList = new int[]{100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000};
		double bestTestAccuracy = 0.0;
		double bestDevAccuracy = 0.0;
		int bestNumFeatures = 0;
		for (int numFeatures : numFeatureList){
			FeatureSelection fs = new FeatureSelection(infoGain, numFeatures); model.data.setFeatureSelection(fs); SimpleConfusionMatrix[] results = model.trainTest(false); SimpleConfusionMatrix devResults = results[0]; SimpleConfusionMatrix testResults = results[1]; 
			double devAccuracy = devResults.getAccuracy();
			double testAccuracy = testResults.getAccuracy();
			if (testAccuracy > bestTestAccuracy) {
				bestNumFeatures = numFeatures;
				bestTestAccuracy = testAccuracy;
				bestDevAccuracy = devAccuracy;
			}
			System.out.println(numFeatures + ", " + devAccuracy + ", " + testAccuracy);
			
			HashMap<String, Object> resultInfo = new HashMap<String, Object>();
			resultInfo.put("data", "dev");
			resultInfo.put("num features", Integer.toString(numFeatures));
			resultInfo.put("accuracy", Double.toString(devAccuracy));
			rw.write(devResults, resultInfo);	
			resultInfo.put("data", "test");
			resultInfo.put("num features", Integer.toString(numFeatures));
			resultInfo.put("accuracy", Double.toString(testAccuracy));
			rw.write(testResults, resultInfo);	
		}
		System.out.println(bestNumFeatures + ", " + bestDevAccuracy + ", " + bestTestAccuracy);
		rw.close();
		
	}
	

	public static void main(String[] args) throws JSONException, IOException {
		// TODO Auto-generated method stub
		new InfoGainFeatureSelectionExp(args[0], args[1]);
	}
}
