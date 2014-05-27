/**
 * Copyright (c) 2013 NearMe, All Rights Reserved.
 * FileName:BytesEncoder.java
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

import org.jboss.netty.buffer.ChannelBuffers;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.handler.codec.oneone.OneToOneEncoder;

/**
 * ClassName:BytesEncoder
 * Function: byte[]类型编码
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2013-8-22  上午11:19:17
 */
public class BytesEncoder extends OneToOneEncoder {
	private boolean copy;
	
	public BytesEncoder() {
		this(false);
	}
	
	public BytesEncoder(boolean copy) {
		super();
		
		this.copy = copy;
	}

	@Override
	protected Object encode(ChannelHandlerContext ctx, Channel channel,
			Object msg) throws Exception {
		if (!(msg instanceof byte[])) {
            return msg;
        }
		
		byte[] msgBytes = (byte[])msg;
		return copy ? ChannelBuffers.copiedBuffer(msgBytes)
				: ChannelBuffers.wrappedBuffer(msgBytes);
	}
}

