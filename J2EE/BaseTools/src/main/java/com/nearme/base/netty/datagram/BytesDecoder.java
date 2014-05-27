/**
 * Copyright (c) 2013 NearMe, All Rights Reserved.
 * FileName:BytesDecoder.java
 * ProjectName:NearmeBaseToolsJ
 * PackageName:com.nearme.base.netty.datagram
 * Create Date:2013-8-22
 * History:
 *   ver	date	  author		desc	
 * ────────────────────────────────────────────────────────
 *   1.0	2013-8-22	  80036381		
 *
 * 
*/

package com.nearme.base.netty.datagram;

import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.handler.codec.oneone.OneToOneDecoder;

import com.nearme.base.netty.common.DataHelper;

/**
 * ClassName:BytesDecoder
 * Function: byte[]解码
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2013-8-22  上午11:25:21
 */
public class BytesDecoder extends OneToOneDecoder {

	@Override
	protected Object decode(ChannelHandlerContext ctx, Channel ch,
			Object msg) throws Exception {
		if (!(msg instanceof ChannelBuffer)) {
            return msg;
        }

		ChannelBuffer cb = (ChannelBuffer) msg;
        return DataHelper.getBytesFromChannelBuffer(cb);
	}
}

