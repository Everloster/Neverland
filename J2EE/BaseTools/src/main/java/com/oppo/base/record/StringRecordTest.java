/**
 * StringRecordTest.java
 * com.oppo.base.record
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-9-29 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.record;
/**
 * ClassName:StringRecordTest
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-9-29  下午03:28:20
 */
public abstract class StringRecordTest extends AbstractFileRunnableRecord<String> {
//	private static final StringRecordTest instance = new StringRecordTest();

	/**
	 * 为了方便测试，每五条记录一次
	 * @see com.oppo.base.record.AbstractFileRunnableRecord#getMinHandleSize()
	 */
	protected int getMinHandleSize() {
		return 5;
	}
//	
//	@Override
//	protected String getSavePath() {
//		//30分钟换一个文件
//		long time = System.currentTimeMillis() / 1800000;
//		return String.format("G:\\String%d.txt", time);
//	}

	@Override
	protected String getString(String tObj) {
		return tObj;
	}

	@Override
	protected void error(String info, Throwable throwable) {
		System.out.println(info);
		
		throwable.printStackTrace();
	}
	
	public static void main(String[] args) {
		//插入10行
		StringRecordTest stringRecordTest = new StringRecordTest() {
			@Override
			protected String getSavePath() {
				long time = System.currentTimeMillis() / 1800000;
				return String.format("G:\\String%d.txt", time);
			}
		};
		
		stringRecordTest.addRecordTask("a");
		stringRecordTest.addRecordTask("b");
		stringRecordTest.addRecordTask("v");
		stringRecordTest.addRecordTask("d");
		stringRecordTest.addRecordTask("e");
		stringRecordTest.addRecordTask("f");
		stringRecordTest.addRecordTask("g");
		stringRecordTest.addRecordTask("h");
		stringRecordTest.addRecordTask("i");
		stringRecordTest.addRecordTask("j");
		
		//
		try {
			Thread.sleep(15000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
}

