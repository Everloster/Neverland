/**
 * IBusinessFactory.java
 * com.nearme.base.netty.server
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-5-31 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.server;

import java.io.InputStream;

/**
 * ClassName:IBusinessFactory <br>
 * Function: 业务处理工厂，提供注册与获取处理器的功能 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-31  下午5:40:13
 */
public interface IBusinessFactory {
	/**
	 * 使用xml格式的配置文件注册
	 * @param
	 * @return
	 */
	boolean registerXml(InputStream inputStream);

	/**
	 * 将业务类型及对应的处理模块进行注册
	 * @param
	 * @return
	 */
	void register(int businessType, IBusinessHandler handler);

	/**
	 * 根据业务类型获取对应的处理模块
	 * @param
	 * @return
	 */
	IBusinessHandler getBusinessHandler(int businessType);
}

