/**
 * DefaultSerializeMessageTest.java
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

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.Header;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.RequestData;

/**
 * ClassName:DefaultSerializeMessageTest <br>
 * Function: TODO ADD FUNCTION <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-16  下午5:15:54
 */
public class DefaultSerializeMessageTest {
	@Test
	public void testEmptyMessage() {
		DefaultLazyMessage msg = new DefaultLazyMessage();
		msg.toByteArray();
	}

	@Test
	public void testGenerateSerializeMessage() {
		DefaultLazyMessage msg = new DefaultLazyMessage();
		//int添加
		msg.setInt(1, 500);
		//long添加
		msg.setLong(2, 123123);
		//float添加
		msg.setFloat(3, .3f);
		//double添加
		msg.setDouble(4, 5.5);
		//boolean添加
		msg.setBoolean(5, true);
		//string添加
		msg.setString(6, "outter中文");
		//字节数组添加
		msg.setBytes(7, new byte[]{ 1, 2});

		//message添加
		DefaultLazyMessage msg0 = new DefaultLazyMessage();
		msg0.setInt(1, 500);
		msg0.setLong(2, 123123);
		msg0.setFloat(3, .3f);
		msg0.setDouble(4, 5.5);
		msg0.setBoolean(5, true);
		msg0.setString(6, "innner中文");
		msg0.setBytes(7, new byte[]{ 1, 2});
		msg.setMessage(8, msg0);

		//int数组
		msg.addIntToList(9, 1);
		msg.addIntToList(9, 3);
		//long数组
		msg.addLongToList(10, 10);
		msg.addLongToList(10, 11);
		//float
		msg.addFloatToList(11, 11.0f);
		msg.addFloatToList(11, 11.1f);
		//double
		msg.addDoubleToList(12, 12.0);
		msg.addDoubleToList(12, 12.1);
		//bool
		msg.addBooleanToList(13, true);
		msg.addBooleanToList(13, false);
		//string
		msg.addStringToList(14, "from 14-1");
		msg.addStringToList(14, "from 14-2");
		//bytes
		msg.addBytesToList(15, new byte[]{1});
		msg.addBytesToList(15, new byte[]{2});
		//message
		msg.addMessageToList(16, msg0);

		DefaultLazyMessage msg1 = new DefaultLazyMessage();
		msg1.setMetaData(msg0.getMetaData());
		msg1.setInt(1, 501);
		msg1.setLong(2, 123113);
		msg1.setFloat(3, .31f);
		msg1.setDouble(4, 5.15);
		msg1.setBoolean(5, true);
		msg1.setString(6, "innner中文2");
		msg1.setBytes(7, new byte[]{ 1, 2});
		msg.addMessageToList(16, msg1);

		//System.out.println(msg.toString());


		try {
			byte[] data = msg.toByteArray();
			long s = System.currentTimeMillis();
			for(int i = 0; i < 1000000; i++) {
				msg.toByteArray();
			}
			long t = System.currentTimeMillis() - s;
			System.out.println(data.length + ",编码耗时：" + t);

			DefaultLazyMessage parseMsg = new DefaultLazyMessage();
			parseMsg.mergeFrom(data);
			//System.out.println(parseMsg);

			s = System.currentTimeMillis();
			for (int i = 0; i < 1000000; i++) {
				DefaultLazyMessage parseMsg1 = new DefaultLazyMessage();
				parseMsg1.mergeFrom(data);
			}
			t = System.currentTimeMillis() - s;
			System.out.println("解析耗时：" + t);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			TestCase.fail();
		}

		TestCase.assertTrue(true);
	}

	@Test
	public void testEncode() {
		int size = 1000000;
		computeLazy(100);
		long s = System.currentTimeMillis();
		computeLazy(size);
		long t = System.currentTimeMillis() - s;
		System.out.println("循环次数：" + size + ",lazy:" + t);

		computePb(100);
		s = System.currentTimeMillis();
		computePb(size);
		t = System.currentTimeMillis() - s;

		System.out.println("循环次数：" + size + ",pb:" + t);

//		DefaultLazyMessage lazy = getLazyMsg();
//		System.out.println(lazy.toByteArray().length);
//
//		RequestData pb = getPb();
//		System.out.println(pb.toByteArray().length);
	}

	@Test
	public void testDecode() {
		byte[] lazy = getLazyMsg().toByteArray();
		byte[] pb = getPb().toByteArray();
		decodeLazy(lazy, 100);
		decodePb(pb, 100);

		int size = 1000000;
		long s = System.currentTimeMillis();
		decodeLazy(lazy, size);
		long t = System.currentTimeMillis() - s;
		System.out.println("循环次数：" + size + ",lazy:" + (t*1.0/size));

		s = System.currentTimeMillis();
		decodePb(pb, size);
		t = System.currentTimeMillis() - s;

		System.out.println("循环次数：" + size + ",pb:" + (t*1.0/size));
	}

	private void decodeLazy(byte[] data, int times) {
		for (int i = 0; i < times; i++) {
			DefaultLazyMessage lazy = new DefaultLazyMessage();
			try {
				lazy.mergeFrom(data);
			} catch (IOException e) {

				// TODO Auto-generated catch block
				e.printStackTrace();

			}
		}
	}

	private void decodePb(byte[] data, int times) {
		for (int i = 0; i < times; i++) {
			//DefaultLazyMessage lazy = new DefaultLazyMessage();
			try {
				RequestData.parseFrom(data);
			} catch (InvalidProtocolBufferException e) {

				// TODO Auto-generated catch block
				e.printStackTrace();

			}
		}
	}

	private void computeLazy(int times) {
		for (int i = 0; i < times; i++) {
			DefaultLazyMessage lazy = getLazyMsg();
			lazy.toByteArray();
		}
	}

	private void computePb(int times) {
		for (int i = 0; i < times; i++) {
			RequestData pb = getPb();
			pb.toByteArray();
		}
	}

	public DefaultLazyMessage getLazyMsg() {
		DefaultLazyMessage lazy = new DefaultLazyMessage();
		lazy.setString(1, "272.16.12.111");
		lazy.setInt(2, 100);
		lazy.setInt(3, 20000);
		lazy.setLong(4, 1000000);
		lazy.setBytes(5, getBytes());

		DefaultLazyMessage l0 = new DefaultLazyMessage();
		l0.setString(1, "Ext-System");
		l0.setString(2, "X905/2.3.6/2/2/2/V1.8/1");
		lazy.addMessageToList(6, l0);

		DefaultLazyMessage l1 = new DefaultLazyMessage();
		l1.setMetaData(l0.getMetaData());
		l1.setString(1, "VersionCode");
		l1.setString(2, "180");
		lazy.addMessageToList(6, l1);

		DefaultLazyMessage l2 = new DefaultLazyMessage();
		l2.setMetaData(l0.getMetaData());
		l2.setString(1, "Ext-User");
		l2.setString(2, "-1/860839019998418/0");
		lazy.addMessageToList(6, l2);

		DefaultLazyMessage l3 = new DefaultLazyMessage();
		l3.setMetaData(l0.getMetaData());
		l3.setString(1, "Content-Type");
		l3.setString(2, "application/octet-stream");
		lazy.addMessageToList(6, l3);

		return lazy;
	}

	public RequestData getPb() {
		RequestData.Builder pbBuilder = RequestData.newBuilder();
		pbBuilder.setClientIp("272.16.12.111");
		pbBuilder.setRequestType(100);
		pbBuilder.setTimeout(20000);
		pbBuilder.setId(1000000);
		pbBuilder.setRequestData(ByteString.copyFrom(getBytes()));

		Header.Builder h = Header.newBuilder();
		h.setKey("Ext-System");
		h.setValue("X905/2.3.6/2/2/2/V1.8/1");
		pbBuilder.addHeader(h.build());

		h.setKey("VersionCode");
		h.setValue("180");
		pbBuilder.addHeader(h.build());

		h.setKey("Ext-User");
		h.setValue("-1/860839019998418/0");
		pbBuilder.addHeader(h.build());

		h.setKey("Content-Type");
		h.setValue("application/octet-stream");
		pbBuilder.addHeader(h.build());
		return pbBuilder.build();
	}

	private byte[] getBytes() {
		return new byte[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 23,23,22};
	}
}

