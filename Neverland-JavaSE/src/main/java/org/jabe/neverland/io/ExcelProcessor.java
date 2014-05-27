package org.jabe.neverland.io;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;

import jxl.Sheet;
import jxl.Workbook;

public class ExcelProcessor {
	public static final String N = "\r\n";
	public static void main(String[] args) {
		try {
			File file = new File("H:/mz.xls");
			InputStream is = new FileInputStream(file);
			File outFile = new File("H:/mz.xml");
			outFile.createNewFile();
			FileOutputStream fileOutputStream = new FileOutputStream(outFile);
			Workbook wb = Workbook.getWorkbook(is);
			Sheet sheets[] = wb.getSheets();
			Sheet sheet1 = sheets[0];// 266 715
			System.out.println(sheet1.getColumns() + ":" + sheet1.getRows());
			for (int i = 1; i < 266; i++) {
				String temp = "<item>" + sheet1.getCell(0, i).getContents() + "</item>";
				fileOutputStream.write(temp.getBytes());
				fileOutputStream.write(N.getBytes());
			}
			fileOutputStream.write("~~~~~~~~~~~~~~".getBytes());
			for (int i = 266; i < sheet1.getRows(); i++) {
				String temp = "<item>" + sheet1.getCell(0, i).getContents() + "</item>";
				fileOutputStream.write(temp.getBytes());
				fileOutputStream.write(N.getBytes());
			}
		} catch (Exception e) {
		}
	}
}
