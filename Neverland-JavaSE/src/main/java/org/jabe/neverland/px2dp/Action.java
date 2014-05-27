package org.jabe.neverland.px2dp;

import java.awt.Desktop;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JOptionPane;

public class Action {
	public static final float SCALE = 3.0F;
	
	public static synchronized void ExeLaout(String inPath, String outPath,
			JButton jb, JFrame jf) {
		jb.setEnabled(false);
		if (!inPath.endsWith(File.separator)) {
			inPath = inPath + File.separator;
		}
		if (!outPath.endsWith(File.separator))
			outPath = outPath + File.separator;
		try {
			File f = new File(inPath);
			String[] list = f.list();
			for (String string : list) {
				if (!string.endsWith("xml")) {
					continue;
				}
				File file = new File(inPath + string);
				File oFile = new File(outPath + string);
				oFile.createNewFile();
				FileWriter fw = new FileWriter(oFile);

				BufferedReader input = new BufferedReader(new FileReader(file));
				StringBuffer buffer = new StringBuffer();
				String text;
				while ((text = input.readLine()) != null) {
					buffer.append(text + "\n");
				}
				String toString = buffer.toString();
				Pattern p = Pattern.compile("\"(\\d+)px\"");

				Matcher m = p.matcher(toString);
				while (m.find()) {
					String group = m.group();
					group = group.replaceAll("\"", "");
					group = group.replaceAll("px", "");
					float dI = px2dip(Float.parseFloat(group));
					group = "\"" + dI + "dp\"";
					toString = toString.replace(m.group(), group);
				}

				fw.write(toString);
				fw.flush();
				fw.close();
			}

			JOptionPane.showMessageDialog(jf.getContentPane(), "操作成功");
			Desktop.getDesktop().open(new File(outPath));
		} catch (Exception ex) {
			JOptionPane.showMessageDialog(jf.getContentPane(), "操作失败");
		} finally {
			jb.setEnabled(true);
		}
	}
	
	public static final String DP = "dp";
	public static String SP = "sp";

	public static synchronized void ExeStyle(String inPath, String outPath,
			JButton jb, JFrame jf) {
		jb.setEnabled(false);
		if (!inPath.endsWith(File.separator)) {
			inPath = inPath + File.separator;
		}
		if (!outPath.endsWith(File.separator))
			outPath = outPath + File.separator;
		try {
			File f = new File(inPath);
			String[] list = f.list();
			for (String string : list) {
				String ending = string.contains("text") ? SP : DP;
				if (!string.endsWith("xml")) {
					continue;
				}
				File file = new File(inPath + string);
				File oFile = new File(outPath + string);
				oFile.createNewFile();
				FileWriter fw = new FileWriter(oFile);

				BufferedReader input = new BufferedReader(new FileReader(file));
				StringBuffer buffer = new StringBuffer();
				String text;
				Pattern p = Pattern.compile("(\\d+)px");
				
				while ((text = input.readLine()) != null) {
					Matcher m = p.matcher(text);		
					while (m.find()) {
						String old = m.group();
						String newS = old.replaceAll("px", "");
						float dI = px2dip(Float.parseFloat(newS));
						DecimalFormat df = new DecimalFormat("###.##");
						String newEnding = df.format(new Float(dI).doubleValue());
						newS = newEnding + ending;
						text = text.replace(old, newS);
					}
					buffer.append(text + "\n");
					
				}
				String toString = buffer.toString();
				fw.write(toString);
				fw.flush();
				fw.close();
			}

			JOptionPane.showMessageDialog(jf.getContentPane(), "操作成功");
			Desktop.getDesktop().open(new File(outPath));
		} catch (Exception e) {
			JOptionPane.showMessageDialog(jf.getContentPane(), "操作失败");
		} finally {
			jb.setEnabled(true);
		}
	}

	private static float px2dip(float pxValue) {
		return (pxValue / SCALE + 0.5F);
	}
	
	private static int dip2px(float dpValue) {
		return (int) (dpValue * SCALE + 0.5F);
	}
}
