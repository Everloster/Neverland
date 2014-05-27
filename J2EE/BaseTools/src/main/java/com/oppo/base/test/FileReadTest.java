/**
 * FileReadTest.java
 * com.oppo.base.test
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-21 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.test;

import java.io.File;
import java.io.IOException;

import com.nearme.base.file.FileHelper;
import com.oppo.base.file.FileOperate;

/**
 * ClassName:FileReadTest
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-21  下午01:53:22
 */
public class FileReadTest {
	public static void main(String[] args) {
		for (int i = 1; i < 5; i++) {  
			ConcurrentTestUtil.compare(10 * i, 100 * i, 
					new OIOReadTest(),	//速度排名3
					new NIOReadTest(),	//速度排名2
					new OPPOIOReadTest()	//速度排名1
			);  	
        }
	}
	
	public static File getReadFile() {
		return new File("G:\\uptbas.sql");
	}
}

class OIOReadTest extends ConcurrentTest {
	public OIOReadTest() {
		super("old io read");
	}

	@Override
	protected void test() {
		try {
			FileOperate.getFileContent(FileReadTest.getReadFile(), null);
		} catch (IOException e) {
			
			e.printStackTrace();
		}
	}
	
}

class NIOReadTest extends ConcurrentTest {
	public NIOReadTest() {
		super("new io read");
	}
	
	@Override
	protected void test() {
		try {
			FileOperate.getShareContent(FileReadTest.getReadFile(), null);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}

class OPPOIOReadTest extends ConcurrentTest {
	public OPPOIOReadTest() {
		super("oppo io read");
	}
	
	@Override
	protected void test() {
		try {
			FileHelper.getFileBytes(FileReadTest.getReadFile());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}

