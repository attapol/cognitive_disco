package types;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

import org.json.JSONException;
import org.json.JSONObject;

public class ResultWriter {

	private PrintWriter writer;
	private PrintWriter logWriter;
	private int numResults;
	private StringBuilder log = new StringBuilder();
	
	public ResultWriter(String fileName) throws IOException{
		writer = new PrintWriter(new File(fileName + ".json"));
		logWriter = new PrintWriter(new File(fileName + ".log"));
	}
	
	public ResultWriter(OutputStream out) {
		writer = new PrintWriter(out);
	}
	
	public void logln(String text) {
		logWriter.write(text + "\n");
		log.append(text + "\n");
	}
	
	public void writeAccuracy(SimpleConfusionMatrix cm) throws JSONException {
		double accuracy = cm.computeAccuracy();
		double baselineAccuracy = cm.computeBaselineAccuracy();
		HashMap<String, Object> information = new HashMap<String, Object>();
		information.put("accuracy", accuracy);
		information.put("baseline accuracy", baselineAccuracy);
		write(information);

	}
	
	public void write(SimpleConfusionMatrix cm) throws JSONException {
		write(cm, new HashMap<String, Object>());
	}

	public void write(Map<String, Object> extraInformation) {
		JSONObject json = new JSONObject(extraInformation);
		if (numResults > 0) writer.write("\n");
		writer.write(json.toString());
		numResults++;
	}
	
	public void write(SimpleConfusionMatrix cm, Map<String, Object> extraInformation) throws JSONException{
		JSONObject json = cm.toJson();
		for (String key: extraInformation.keySet()){
			json.put(key, extraInformation.get(key));
		}
		if (numResults > 0) writer.write("\n");
		writer.write(json.toString());
		numResults++;
	}

	public void close() throws JSONException {
		JSONObject json = new JSONObject();
		json.put("log", log.toString());
		writer.write("\n");
		writer.write(json.toString());
		writer.close();
		logWriter.close();
	}
}
