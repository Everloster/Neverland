/**
 * SerializeMetaDataTest.java
 * com.nearme.base.serializer
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-5-16 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.serializer;

import java.io.IOException;

import junit.framework.TestCase;

import org.junit.Test;

/**
 * ClassName:SerializeMetaDataTest <br>
 * Function: TODO ADD FUNCTION <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-16  下午12:11:20
 */
public class SerializeMetaDataTest {

	@Test
	public void testGenerateMetaData() {
		DefaultLazyMetaData metaData = new DefaultLazyMetaData();
		for (int i = 1; i < 50; i++) {
			metaData.setColumnType(i, LazySerializeTypeHelper.getFromType(i % 9 + 1));
		}

		System.out.println(metaData.toString());
		byte[] data = metaData.toByteArray();
		System.out.println(data.length);

		try {
			DefaultLazyMetaData meta = new DefaultLazyMetaData();
			meta.mergeFrom(data);
			System.out.println(meta.toString());

			TestCase.assertTrue(true);
		} catch (IOException e) {
			e.printStackTrace();
			TestCase.fail(e.getMessage());
		}
	}

}

