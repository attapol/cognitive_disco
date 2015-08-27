/**
 * 
 */
package preprocess;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.json.JSONException;
import org.json.JSONObject;

import cc.mallet.classify.Classification;
import cc.mallet.classify.MaxEnt;
import cc.mallet.classify.MaxEntTrainer;
import cc.mallet.types.Instance;
import types.DataTriplet;
import types.LabelType;

/**
 * This will train the models and cache the classification results
 * onto the JSON data files. These will be useful for Mixture of Experts model
 * that we will use in a neural network architecture.
 * @author te
 *
 */
public class CacheClassificationAnnotator {

	/**
	 * @return 
	 * @throws JSONException 
	 * @throws IOException 
	 *  
	 */
	public CacheClassificationAnnotator(String experimentName, String trainingDir, String devDir, String testDir) throws IOException, JSONException {
		DataTriplet dataTriplet = new DataTriplet(
				trainingDir + "/" + experimentName + ".original_label.features",
				devDir + "/" + experimentName + ".original_label.features",
				testDir + "/" + experimentName + ".original_label.features");
		dataTriplet.importData(LabelType.SCHEME_B);
		MaxEntTrainer trainer = new MaxEntTrainer();
		trainer.train(dataTriplet.getTrainingSet());
		MaxEnt classifier = trainer.getClassifier();
		annotate(trainingDir + "/" + "pdtb-data-plus.json", classifier.classify(dataTriplet.getTrainingSet()));
		annotate(devDir + "/" + "pdtb-data-plus.json", classifier.classify(dataTriplet.getDevSet()));
		annotate(testDir + "/" + "pdtb-data-plus.json", classifier.classify(dataTriplet.getTestSet()));

	}
	
	public void annotate(String fileName, ArrayList<Classification> classification) throws IOException, JSONException {
		HashMap<String, double[]> id2label = new HashMap<String, double[]>(); 
		for (Classification c : classification){
			Instance instance = c.getInstance();
			String name = (String) instance.getName();
			//String predictedLabel = (String)instance.getLabeling().getBestLabel().getEntry();
			//int numLabels = c.getLabelVector().getNumDimensions();
			int numLabels = c.getLabelVector().getLabelAlphabet().size();
			//System.out.println(numLabels);
			double[] scores = new double[numLabels];
			c.getLabelVector().arrayCopyInto(scores, 0);
			id2label.put(name, scores);
		}

		List<String> relations = FileUtils.readLines(new File(fileName));
		FileWriter wr = new FileWriter(new File(fileName));	
		for (String relation : relations){
			JSONObject relationJSON = new JSONObject(relation);
			String key = relationJSON.getString("DocID") + "_" + relationJSON.getInt("ID");
			double[] scores = id2label.get(key);
			if (scores != null) {
                for (int i = 0; i < scores.length; i++) {
                        scores[i] = Math.round(scores[i] * 10000) / 10000.0;
                }
				relationJSON.put("BaselineClassification", scores);
			}
			relationJSON.write(wr);
			wr.write("\n");
		}
		wr.close();
	}
	

	/**
	 * @param args
	 * @throws JSONException 
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException, JSONException {
		// TODO Auto-generated method stub
		CacheClassificationAnnotator classifier = new CacheClassificationAnnotator(args[0], "conll15-st-05-19-15-train", "conll15-st-05-19-15-dev", "conll15-st-05-19-15-test");
	}

}
