package org.jabe.neverland.oauth;

import java.io.RandomAccessFile;
import java.text.DecimalFormat;

import com.nearme.oauth.model.AccessToken;
import com.nearme.oauth.open.AccountAgent;
import com.nearme.oauth.provider.URLProvider;
import com.nearme.oauth.util.Constants;

public class Test {
	public static void main(String[] args) {
		
		URLProvider.init();
		Constants.APP_KEY = "c5217trjnrmU6gO5jG8VvUFU0";
		Constants.APP_SECRET = "e2eCa732422245E8891F6555e999878B";
		AccessToken at = new AccessToken("4ee3e0757a4688e3c3255a39cbdf59f4", "9148de7c78ce2aba37b0b59ea5fa495c");
		try {
			String gcUserInfo = AccountAgent.getInstance().getGCUserInfo(at);
			System.out.println(gcUserInfo);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	RandomAccessFile ra ;
}
