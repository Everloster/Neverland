/**
 * NettyBusinessFactory.java
 * com.nearme.base.netty.server
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-4-26 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.server;

import java.io.InputStream;

/**
 * ClassName:NettyBusinessFactory <br>
 * Function: 业务处理工厂类 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-4-26  下午11:51:49
 */
public final class NettyBusinessFactory {

	private static final DefaultBusinessFactory defaultBusinessFactory = new DefaultBusinessFactory();

	/**
	 * 获取默认工厂
	 * @param
	 * @return
	 */
	public static IBusinessFactory getDefaultBusinessFactory() {
		return defaultBusinessFactory;
	}

	/**
	 * 使用xml格式的配置文件注册
	 * @param
	 * @return
	 */
	public static boolean registerXml(InputStream inputStream) {
		return defaultBusinessFactory.registerXml(inputStream);
	}

	/**
	 * 将业务类型及对应的处理模块进行注册
	 * @param
	 * @return
	 */
	public static void register(int businessType, IBusinessHandler handler) {
		defaultBusinessFactory.register(businessType, handler);
	}

	/**
	 * 根据业务类型获取对应的处理模块
	 * @param
	 * @return
	 */
	public static IBusinessHandler getBusinessHandler(int businessType) {
		return defaultBusinessFactory.getBusinessHandler(businessType);
	}
}

