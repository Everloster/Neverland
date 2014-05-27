package org.jabe.neverland.io;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class CountTest {
	private static final String PROJECT_DIR = "E:\\WorkspaceForEclipseNew";
	private static final String PROJECT_DIR1 = "E:\\WorkspaceForEclipse\\GameCenter-V2.1";
	private static final String PROJECT_DIR2 = "E:\\WorkspaceForEclipse\\DataExchange";
	private static final String PROJECT_DIR3 = "E:\\WorkspaceForEclipse\\OpenSDK";
	private static final String PROJECT_DIR4 = "E:\\WorkspaceForEclipse\\OpenSDKDemo";
	private static final String PROJECT_DIR5 = "E:\\WorkspaceForEclipse\\OpenOauth";
	private static final String PROJECT_DIR6 = "E:\\WorkspaceForEclipse\\GameCenterSDK_2.0";
	private static final String PROJECT_DIR7 = "E:\\WorkspaceForEclipse\\Oppo_lianliankan2";
	private static final String PROJECT_DIR8 = "E:\\WorkspaceForEclipseNew\\GameCenter-Main";
	private static final String PROJECT_DIR9 = "E:\\WorkspaceForEclipse\\Note";
	
	private static int fileNum = 0;
	private static int lineNum = 0;

	private static void listNext(File dir) {
		File[] files = dir.listFiles();
		for (int i = 0; i < files.length; i++) {
			if (files[i].isDirectory()) {
				if (files[i].getName().equals("bin")) {
					continue;
				}
				listNext(files[i]);
			} else {
				try {
					String[] filter = new String[] {".java", ".xml", ".css", ".js", ".html"};
					if (check(filter, files[i])) {
						fileNum++;
						BufferedReader br = new BufferedReader(new FileReader(
								files[i]));
						while (br.readLine() != null) {
							lineNum++;
						}
					}
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	private static boolean check(String[] strings, File file) {
		for (int i = 0; i < strings.length; i++) {
			String string = strings[i];
			if (file.getName().endsWith(string)) {
				return true;
			}
		}
		return false;
	}

	public static void main(String[] args) throws Exception {
		File root1 = new File(PROJECT_DIR1);
		File root2 = new File(PROJECT_DIR2);
		File root3 = new File(PROJECT_DIR3);
		File root4 = new File(PROJECT_DIR4);
		File root5 = new File(PROJECT_DIR5);
		File root6 = new File(PROJECT_DIR6);
		File root7 = new File(PROJECT_DIR7);
		File root8 = new File(PROJECT_DIR8);
		File root9 = new File(PROJECT_DIR9);
//		listNext(root1);
//		listNext(root2);
		listNext(root8);
//		listNext(root4);
//		listNext(root5);
//		listNext(root6);
//		listNext(root7);
//		File root = new File(PROJECT_DIR4);
//		listNext(root);
		System.out.println("files number: " + fileNum);
		System.out.println("code lines: " + lineNum);
	}

}