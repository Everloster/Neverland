package com.game.u3d.nearme;

import android.os.Bundle;

import com.nearme.gamecenter.open.api.ApiCallback;
import com.nearme.gamecenter.open.api.GameCenterSDK;
import com.nearme.gamecenter.open.api.GameCenterSettings;
import com.nearme.gamecenter.open.api.ProductInfo;
import com.nearme.gamecenter.open.api.UserInfo;
import com.unity3d.player.UnityPlayer;
import com.unity3d.player.UnityPlayerActivity;

public class UnityEntry extends UnityPlayerActivity {

	private final static String GAME_OBJECT_NAME = "GAME_OBJECT_NAME";

	@Override
	protected void onCreate(Bundle arg0) {
		super.onCreate(arg0);
	}

	public void SDK_init() {
		this.runOnUiThread(new Runnable() {
			
			@Override
			public void run() {
				// 测试用的appkey和secret
				GameCenterSettings gameCenterSettings = new GameCenterSettings(
						"c5217trjnrmU6gO5jG8VvUFU0", "e2eCa732422245E8891F6555e999878B") {

					@Override
					public void onForceReLogin() {
						// SDK由于某些原因登出,此方法通知cp,cp需要在此处清理当前的登录状态并重新请求登录.
					}
				};
				GameCenterSettings.isDebugModel = true;// 测试log开关
				GameCenterSettings.isOritationPort = true;// 竖屏
				// GameCenterSettings.isNeedShowLoading = false;
				// GameCenterSettings.request_time_out = 10000;//设置SDK网络
				GameCenterSDK.init(gameCenterSettings, UnityEntry.this);
			}
		});
	}

	public void SDK_doLogin() {
		GameCenterSDK.setmCurrentContext(this);
		runOnUiThread(new Runnable() {
			
			@Override
			public void run() {
				GameCenterSDK.getInstance().doLogin(new ApiCallback() {

					@Override
					public void onSuccess(String content, int code) {
						UnityPlayer.UnitySendMessage("Main", "onNearMeLoginCallback",
								code + "#" + content);
					}

					@Override
					public void onFailure(String content, int code) {
						UnityPlayer.UnitySendMessage("Main", "onNearMeLoginCallback",
								code + "#" + content);
					}
				});
			}
		});
	}

	public void SDK_doGetUserInfo() {
		GameCenterSDK.setmCurrentContext(this);
		GameCenterSDK.getInstance().doGetUserInfo(new ApiCallback() {

			@Override
			public void onSuccess(String content, int code) {
				final UserInfo userInfo = new UserInfo(content);
				final String tokenKey = GameCenterSDK.getInstance()
						.getAccessToken().getToken();
				final String tokenSecret = GameCenterSDK.getInstance()
						.getAccessToken().getSecret();
				sendMessageToUnity(GAME_OBJECT_NAME,
						"onNearMeGetUserInfoOKCallback", code + userInfo.id + tokenKey + tokenSecret);
			}

			@Override
			public void onFailure(String content, int code) {
				sendMessageToUnity(GAME_OBJECT_NAME,
						"onNearMeGetUserInfoFailureCallback", code + "#" + content);
			}
		});
	}

	/**
	 * example : string format :
	 * 100#http://www.mycompany.com/callback#partner_order
	 * #productDesc#productName#pay_type#count
	 * 
	 * @param productInfo
	 */
	public void SDK_doPayment(String productInfo) {
		final String[] data = productInfo.split("#");
		final ProductInfo proInfo = new ProductInfo(Double.valueOf(data[0]),
				data[1], data[2], data[3], data[4], Integer.valueOf(data[5]),
				Integer.valueOf(data[6]));
		// Using orders to divide callback messages
		final String orders = proInfo.getPartnerOrder();
		GameCenterSDK.getInstance().doPayment(new ApiCallback() {

			@Override
			public void onSuccess(String content, int code) {
				sendMessageToUnity(GAME_OBJECT_NAME, "onNearMePaymentCallback",
						code + "#" + orders);
			}

			@Override
			public void onFailure(String content, int code) {
				sendMessageToUnity(GAME_OBJECT_NAME, "onNearMePaymentCallback",
						code + "#" + orders);
			}
		}, proInfo);
	}

	public void SDK_doShowCharge() {
		GameCenterSDK.getInstance().doShowCharge(new ApiCallback() {
			
			@Override
			public void onSuccess(String content, int code) {
				sendMessageToUnity(GAME_OBJECT_NAME, "onNearMeShowChargeCallback", code + "#" + content);
			}
			
			@Override
			public void onFailure(String content, int code) {
				sendMessageToUnity(GAME_OBJECT_NAME, "onNearMeShowChargeCallback", code + "#" + content);
			}
		});
	}
	
	public void SDK_doShowGameCenter() {
		GameCenterSDK.getInstance().doShowGameCenter();
	}
	
	public void SDK_doGetUserBalance() {
		GameCenterSDK.getInstance().doCheckBalance(new ApiCallback() {
			
			@Override
			public void onSuccess(String content, int code) {
				sendMessageToUnity(GAME_OBJECT_NAME, "onNearMeCheckBalanceCallback", code + "#" + content);
			}
			
			@Override
			public void onFailure(String content, int code) {
				sendMessageToUnity(GAME_OBJECT_NAME, "onNearMeCheckBalanceCallback", code + "#" + content);
			}
		});
	}
	
	public void SDK_doReLogin() {
		GameCenterSDK.getInstance().doReLogin(new ApiCallback() {
			
			@Override
			public void onSuccess(String content, int code) {
				sendMessageToUnity(GAME_OBJECT_NAME, "onNearMeReLoginCallback", code + "#" + content);
			}
			
			@Override
			public void onFailure(String content, int code) {
				sendMessageToUnity(GAME_OBJECT_NAME, "onNearMeReLoginCallback", code + "#" + content);
			}
		});
	}

	public void SDK_doShowSprite() {
		GameCenterSDK.setmCurrentContext(this);
		GameCenterSDK.getInstance().doShowSprite(this);
	}

	public void SDK_doDissmissSprite() {
		GameCenterSDK.setmCurrentContext(this);
		GameCenterSDK.getInstance().doDismissSprite(this);
	}

	private void sendMessageToUnity(String gameObjectName, String bindMethod,
			String message) {
		UnityPlayer.UnitySendMessage(gameObjectName, bindMethod, message);
	}

}