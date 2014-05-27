package org.jabe.neverland.plugin;

import java.io.FileDescriptor;
import java.io.PrintWriter;

import android.app.ActionBar;
import android.app.Activity;
import android.app.Fragment;
import android.app.FragmentManager;
import android.app.LoaderManager;
import android.app.PendingIntent;
import android.app.TaskStackBuilder;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentSender;
import android.content.IntentSender.SendIntentException;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.os.Bundle;
import android.util.AttributeSet;
import android.view.ActionMode;
import android.view.ActionMode.Callback;
import android.view.ContextMenu;
import android.view.ContextMenu.ContextMenuInfo;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup.LayoutParams;
import android.view.Window;
import android.view.WindowManager;
import android.view.accessibility.AccessibilityEvent;

public class ActivityWrapper extends Activity {
	
	private Activity mBase;
	
	protected void attachBaseActivity(Activity base) {
        mBase = base;
    }

	@Override
	public Intent getIntent() {
		// TODO Auto-generated method stub
		return mBase.getIntent();
	}

	@Override
	public void setIntent(Intent newIntent) {
		// TODO Auto-generated method stub
		mBase.setIntent(newIntent);
	}

	@Override
	public WindowManager getWindowManager() {
		// TODO Auto-generated method stub
		return mBase.getWindowManager();
	}

	@Override
	public Window getWindow() {
		// TODO Auto-generated method stub
		return mBase.getWindow();
	}

	@Override
	public LoaderManager getLoaderManager() {
		// TODO Auto-generated method stub
		return mBase.getLoaderManager();
	}

	@Override
	public View getCurrentFocus() {
		// TODO Auto-generated method stub
		return mBase.getCurrentFocus();
	}

	@Override
	public boolean onCreateThumbnail(Bitmap outBitmap, Canvas canvas) {
		// TODO Auto-generated method stub
		return mBase.onCreateThumbnail(outBitmap, canvas);
	}

	@Override
	public CharSequence onCreateDescription() {
		// TODO Auto-generated method stub
		return mBase.onCreateDescription();
	}

	@Override
	public void onConfigurationChanged(Configuration newConfig) {
		// TODO Auto-generated method stub
		mBase.onConfigurationChanged(newConfig);
	}

	@Override
	public int getChangingConfigurations() {
		// TODO Auto-generated method stub
		return mBase.getChangingConfigurations();
	}

	@SuppressWarnings("deprecation")
	@Override
	public Object getLastNonConfigurationInstance() {
		// TODO Auto-generated method stub
		return mBase.getLastNonConfigurationInstance();
	}

	@SuppressWarnings("deprecation")
	@Override
	public Object onRetainNonConfigurationInstance() {
		// TODO Auto-generated method stub
		return mBase.onRetainNonConfigurationInstance();
	}

	@Override
	public void onLowMemory() {
		// TODO Auto-generated method stub
		mBase.onLowMemory();
	}

	@Override
	public void onTrimMemory(int level) {
		// TODO Auto-generated method stub
		mBase.onTrimMemory(level);
	}

	@Override
	public FragmentManager getFragmentManager() {
		// TODO Auto-generated method stub
		return mBase.getFragmentManager();
	}

	@Override
	public void onAttachFragment(Fragment fragment) {
		// TODO Auto-generated method stub
		mBase.onAttachFragment(fragment);
	}

	@SuppressWarnings("deprecation")
	@Override
	public void startManagingCursor(Cursor c) {
		// TODO Auto-generated method stub
		mBase.startManagingCursor(c);
	}

	@SuppressWarnings("deprecation")
	@Override
	public void stopManagingCursor(Cursor c) {
		// TODO Auto-generated method stub
		mBase.stopManagingCursor(c);
	}

	@Override
	public View findViewById(int id) {
		// TODO Auto-generated method stub
		return mBase.findViewById(id);
	}

	@Override
	public ActionBar getActionBar() {
		// TODO Auto-generated method stub
		return mBase.getActionBar();
	}

	@Override
	public void setContentView(int layoutResID) {
		// TODO Auto-generated method stub
		mBase.setContentView(layoutResID);
	}

	@Override
	public void setContentView(View view) {
		// TODO Auto-generated method stub
		mBase.setContentView(view);
	}

	@Override
	public void setContentView(View view, LayoutParams params) {
		// TODO Auto-generated method stub
		mBase.setContentView(view, params);
	}

	@Override
	public void addContentView(View view, LayoutParams params) {
		// TODO Auto-generated method stub
		mBase.addContentView(view, params);
	}

	@Override
	public void setFinishOnTouchOutside(boolean finish) {
		// TODO Auto-generated method stub
		mBase.setFinishOnTouchOutside(finish);
	}

	@Override
	public boolean onKeyDown(int keyCode, KeyEvent event) {
		// TODO Auto-generated method stub
		return mBase.onKeyDown(keyCode, event);
	}

	@Override
	public boolean onKeyLongPress(int keyCode, KeyEvent event) {
		// TODO Auto-generated method stub
		return mBase.onKeyLongPress(keyCode, event);
	}

	@Override
	public boolean onKeyUp(int keyCode, KeyEvent event) {
		// TODO Auto-generated method stub
		return mBase.onKeyUp(keyCode, event);
	}

	@Override
	public boolean onKeyMultiple(int keyCode, int repeatCount, KeyEvent event) {
		// TODO Auto-generated method stub
		return mBase.onKeyMultiple(keyCode, repeatCount, event);
	}

	@Override
	public void onBackPressed() {
		// TODO Auto-generated method stub
		mBase.onBackPressed();
	}

	@Override
	public boolean onKeyShortcut(int keyCode, KeyEvent event) {
		// TODO Auto-generated method stub
		return mBase.onKeyShortcut(keyCode, event);
	}

	@Override
	public boolean onTouchEvent(MotionEvent event) {
		// TODO Auto-generated method stub
		return mBase.onTouchEvent(event);
	}

	@Override
	public boolean onTrackballEvent(MotionEvent event) {
		// TODO Auto-generated method stub
		return mBase.onTrackballEvent(event);
	}

	@Override
	public boolean onGenericMotionEvent(MotionEvent event) {
		// TODO Auto-generated method stub
		return mBase.onGenericMotionEvent(event);
	}

	@Override
	public void onUserInteraction() {
		// TODO Auto-generated method stub
		mBase.onUserInteraction();
	}

	@Override
	public void onWindowAttributesChanged(
			android.view.WindowManager.LayoutParams params) {
		// TODO Auto-generated method stub
		mBase.onWindowAttributesChanged(params);
	}

	@Override
	public void onContentChanged() {
		// TODO Auto-generated method stub
		mBase.onContentChanged();
	}

	@Override
	public void onWindowFocusChanged(boolean hasFocus) {
		// TODO Auto-generated method stub
		mBase.onWindowFocusChanged(hasFocus);
	}

	@Override
	public void onAttachedToWindow() {
		// TODO Auto-generated method stub
		mBase.onAttachedToWindow();
	}

	@Override
	public void onDetachedFromWindow() {
		// TODO Auto-generated method stub
		mBase.onDetachedFromWindow();
	}

	@Override
	public boolean hasWindowFocus() {
		// TODO Auto-generated method stub
		return mBase.hasWindowFocus();
	}

	@Override
	public boolean dispatchKeyEvent(KeyEvent event) {
		// TODO Auto-generated method stub
		return mBase.dispatchKeyEvent(event);
	}

	@Override
	public boolean dispatchKeyShortcutEvent(KeyEvent event) {
		// TODO Auto-generated method stub
		return mBase.dispatchKeyShortcutEvent(event);
	}

	@Override
	public boolean dispatchTouchEvent(MotionEvent ev) {
		// TODO Auto-generated method stub
		return mBase.dispatchTouchEvent(ev);
	}

	@Override
	public boolean dispatchTrackballEvent(MotionEvent ev) {
		// TODO Auto-generated method stub
		return mBase.dispatchTrackballEvent(ev);
	}

	@Override
	public boolean dispatchGenericMotionEvent(MotionEvent ev) {
		// TODO Auto-generated method stub
		return mBase.dispatchGenericMotionEvent(ev);
	}

	@Override
	public boolean dispatchPopulateAccessibilityEvent(AccessibilityEvent event) {
		// TODO Auto-generated method stub
		return mBase.dispatchPopulateAccessibilityEvent(event);
	}

	@Override
	public View onCreatePanelView(int featureId) {
		// TODO Auto-generated method stub
		return mBase.onCreatePanelView(featureId);
	}

	@Override
	public boolean onCreatePanelMenu(int featureId, Menu menu) {
		// TODO Auto-generated method stub
		return mBase.onCreatePanelMenu(featureId, menu);
	}

	@Override
	public boolean onPreparePanel(int featureId, View view, Menu menu) {
		// TODO Auto-generated method stub
		return mBase.onPreparePanel(featureId, view, menu);
	}

	@Override
	public boolean onMenuOpened(int featureId, Menu menu) {
		// TODO Auto-generated method stub
		return mBase.onMenuOpened(featureId, menu);
	}

	@Override
	public boolean onMenuItemSelected(int featureId, MenuItem item) {
		// TODO Auto-generated method stub
		return mBase.onMenuItemSelected(featureId, item);
	}

	@Override
	public void onPanelClosed(int featureId, Menu menu) {
		// TODO Auto-generated method stub
		mBase.onPanelClosed(featureId, menu);
	}

	@Override
	public void invalidateOptionsMenu() {
		// TODO Auto-generated method stub
		mBase.invalidateOptionsMenu();
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		// TODO Auto-generated method stub
		return mBase.onCreateOptionsMenu(menu);
	}

	@Override
	public boolean onPrepareOptionsMenu(Menu menu) {
		// TODO Auto-generated method stub
		return mBase.onPrepareOptionsMenu(menu);
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		// TODO Auto-generated method stub
		return mBase.onOptionsItemSelected(item);
	}

	@Override
	public boolean onNavigateUp() {
		// TODO Auto-generated method stub
		return mBase.onNavigateUp();
	}

	@Override
	public boolean onNavigateUpFromChild(Activity child) {
		// TODO Auto-generated method stub
		return mBase.onNavigateUpFromChild(child);
	}

	@Override
	public void onCreateNavigateUpTaskStack(TaskStackBuilder builder) {
		// TODO Auto-generated method stub
		mBase.onCreateNavigateUpTaskStack(builder);
	}

	@Override
	public void onPrepareNavigateUpTaskStack(TaskStackBuilder builder) {
		// TODO Auto-generated method stub
		mBase.onPrepareNavigateUpTaskStack(builder);
	}

	@Override
	public void onOptionsMenuClosed(Menu menu) {
		// TODO Auto-generated method stub
		mBase.onOptionsMenuClosed(menu);
	}

	@Override
	public void openOptionsMenu() {
		// TODO Auto-generated method stub
		mBase.openOptionsMenu();
	}

	@Override
	public void closeOptionsMenu() {
		// TODO Auto-generated method stub
		mBase.closeOptionsMenu();
	}

	@Override
	public void onCreateContextMenu(ContextMenu menu, View v,
			ContextMenuInfo menuInfo) {
		// TODO Auto-generated method stub
		mBase.onCreateContextMenu(menu, v, menuInfo);
	}

	@Override
	public void registerForContextMenu(View view) {
		// TODO Auto-generated method stub
		mBase.registerForContextMenu(view);
	}

	@Override
	public void unregisterForContextMenu(View view) {
		// TODO Auto-generated method stub
		mBase.unregisterForContextMenu(view);
	}

	@Override
	public void openContextMenu(View view) {
		// TODO Auto-generated method stub
		mBase.openContextMenu(view);
	}

	@Override
	public void closeContextMenu() {
		// TODO Auto-generated method stub
		mBase.closeContextMenu();
	}

	@Override
	public boolean onContextItemSelected(MenuItem item) {
		// TODO Auto-generated method stub
		return mBase.onContextItemSelected(item);
	}

	@Override
	public void onContextMenuClosed(Menu menu) {
		// TODO Auto-generated method stub
		mBase.onContextMenuClosed(menu);
	}

	@Override
	public boolean onSearchRequested() {
		// TODO Auto-generated method stub
		return mBase.onSearchRequested();
	}

	@Override
	public void startSearch(String initialQuery, boolean selectInitialQuery,
			Bundle appSearchData, boolean globalSearch) {
		// TODO Auto-generated method stub
		mBase.startSearch(initialQuery, selectInitialQuery, appSearchData, globalSearch);
	}

	@Override
	public void triggerSearch(String query, Bundle appSearchData) {
		// TODO Auto-generated method stub
		mBase.triggerSearch(query, appSearchData);
	}

	@Override
	public void takeKeyEvents(boolean get) {
		// TODO Auto-generated method stub
		mBase.takeKeyEvents(get);
	}

	@Override
	public LayoutInflater getLayoutInflater() {
		// TODO Auto-generated method stub
		return mBase.getLayoutInflater();
	}

	@Override
	public MenuInflater getMenuInflater() {
		// TODO Auto-generated method stub
		return mBase.getMenuInflater();
	}


	@Override
	public void startActivityForResult(Intent intent, int requestCode) {
		// TODO Auto-generated method stub
		mBase.startActivityForResult(intent, requestCode);
	}

	@Override
	public void startActivityForResult(Intent intent, int requestCode,
			Bundle options) {
		// TODO Auto-generated method stub
		mBase.startActivityForResult(intent, requestCode, options);
	}

	@Override
	public void startIntentSenderForResult(IntentSender intent,
			int requestCode, Intent fillInIntent, int flagsMask,
			int flagsValues, int extraFlags) throws SendIntentException {
		// TODO Auto-generated method stub
		mBase.startIntentSenderForResult(intent, requestCode, fillInIntent, flagsMask,
				flagsValues, extraFlags);
	}

	@Override
	public void startIntentSenderForResult(IntentSender intent,
			int requestCode, Intent fillInIntent, int flagsMask,
			int flagsValues, int extraFlags, Bundle options)
			throws SendIntentException {
		// TODO Auto-generated method stub
		mBase.startIntentSenderForResult(intent, requestCode, fillInIntent, flagsMask,
				flagsValues, extraFlags, options);
	}

	@Override
	public void startActivity(Intent intent) {
		// TODO Auto-generated method stub
		mBase.startActivity(intent);
	}

	@Override
	public void startActivity(Intent intent, Bundle options) {
		// TODO Auto-generated method stub
		mBase.startActivity(intent, options);
	}

	@Override
	public void startActivities(Intent[] intents) {
		// TODO Auto-generated method stub
		mBase.startActivities(intents);
	}

	@Override
	public void startActivities(Intent[] intents, Bundle options) {
		// TODO Auto-generated method stub
		mBase.startActivities(intents, options);
	}

	@Override
	public void startIntentSender(IntentSender intent, Intent fillInIntent,
			int flagsMask, int flagsValues, int extraFlags)
			throws SendIntentException {
		// TODO Auto-generated method stub
		mBase.startIntentSender(intent, fillInIntent, flagsMask, flagsValues,
				extraFlags);
	}

	@Override
	public void startIntentSender(IntentSender intent, Intent fillInIntent,
			int flagsMask, int flagsValues, int extraFlags, Bundle options)
			throws SendIntentException {
		// TODO Auto-generated method stub
		mBase.startIntentSender(intent, fillInIntent, flagsMask, flagsValues,
				extraFlags, options);
	}

	@Override
	public boolean startActivityIfNeeded(Intent intent, int requestCode) {
		// TODO Auto-generated method stub
		return mBase.startActivityIfNeeded(intent, requestCode);
	}

	@Override
	public boolean startActivityIfNeeded(Intent intent, int requestCode,
			Bundle options) {
		// TODO Auto-generated method stub
		return mBase.startActivityIfNeeded(intent, requestCode, options);
	}

	@Override
	public boolean startNextMatchingActivity(Intent intent) {
		// TODO Auto-generated method stub
		return mBase.startNextMatchingActivity(intent);
	}

	@Override
	public boolean startNextMatchingActivity(Intent intent, Bundle options) {
		// TODO Auto-generated method stub
		return mBase.startNextMatchingActivity(intent, options);
	}

	@Override
	public void startActivityFromChild(Activity child, Intent intent,
			int requestCode) {
		// TODO Auto-generated method stub
		mBase.startActivityFromChild(child, intent, requestCode);
	}

	@Override
	public void startActivityFromChild(Activity child, Intent intent,
			int requestCode, Bundle options) {
		// TODO Auto-generated method stub
		mBase.startActivityFromChild(child, intent, requestCode, options);
	}

	@Override
	public void startActivityFromFragment(Fragment fragment, Intent intent,
			int requestCode) {
		// TODO Auto-generated method stub
		mBase.startActivityFromFragment(fragment, intent, requestCode);
	}

	@Override
	public void startActivityFromFragment(Fragment fragment, Intent intent,
			int requestCode, Bundle options) {
		// TODO Auto-generated method stub
		mBase.startActivityFromFragment(fragment, intent, requestCode, options);
	}

	@Override
	public void startIntentSenderFromChild(Activity child, IntentSender intent,
			int requestCode, Intent fillInIntent, int flagsMask,
			int flagsValues, int extraFlags) throws SendIntentException {
		// TODO Auto-generated method stub
		mBase.startIntentSenderFromChild(child, intent, requestCode, fillInIntent,
				flagsMask, flagsValues, extraFlags);
	}

	@Override
	public void startIntentSenderFromChild(Activity child, IntentSender intent,
			int requestCode, Intent fillInIntent, int flagsMask,
			int flagsValues, int extraFlags, Bundle options)
			throws SendIntentException {
		// TODO Auto-generated method stub
		mBase.startIntentSenderFromChild(child, intent, requestCode, fillInIntent,
				flagsMask, flagsValues, extraFlags, options);
	}

	@Override
	public void overridePendingTransition(int enterAnim, int exitAnim) {
		// TODO Auto-generated method stub
		mBase.overridePendingTransition(enterAnim, exitAnim);
	}

	@Override
	public String getCallingPackage() {
		// TODO Auto-generated method stub
		return mBase.getCallingPackage();
	}

	@Override
	public ComponentName getCallingActivity() {
		// TODO Auto-generated method stub
		return mBase.getCallingActivity();
	}

	@Override
	public void setVisible(boolean visible) {
		// TODO Auto-generated method stub
		mBase.setVisible(visible);
	}

	@Override
	public boolean isFinishing() {
		// TODO Auto-generated method stub
		return mBase.isFinishing();
	}

	@Override
	public boolean isDestroyed() {
		// TODO Auto-generated method stub
		return mBase.isDestroyed();
	}

	@Override
	public boolean isChangingConfigurations() {
		// TODO Auto-generated method stub
		return mBase.isChangingConfigurations();
	}

	@Override
	public void recreate() {
		// TODO Auto-generated method stub
		mBase.recreate();
	}

	@Override
	public void finish() {
		// TODO Auto-generated method stub
		mBase.finish();
	}

	@Override
	public void finishAffinity() {
		// TODO Auto-generated method stub
		mBase.finishAffinity();
	}

	@Override
	public void finishFromChild(Activity child) {
		// TODO Auto-generated method stub
		mBase.finishFromChild(child);
	}

	@Override
	public void finishActivity(int requestCode) {
		// TODO Auto-generated method stub
		mBase.finishActivity(requestCode);
	}

	@Override
	public void finishActivityFromChild(Activity child, int requestCode) {
		// TODO Auto-generated method stub
		mBase.finishActivityFromChild(child, requestCode);
	}


	@Override
	public PendingIntent createPendingResult(int requestCode, Intent data,
			int flags) {
		// TODO Auto-generated method stub
		return mBase.createPendingResult(requestCode, data, flags);
	}

	@Override
	public void setRequestedOrientation(int requestedOrientation) {
		// TODO Auto-generated method stub
		mBase.setRequestedOrientation(requestedOrientation);
	}

	@Override
	public int getRequestedOrientation() {
		// TODO Auto-generated method stub
		return mBase.getRequestedOrientation();
	}

	@Override
	public int getTaskId() {
		// TODO Auto-generated method stub
		return mBase.getTaskId();
	}

	@Override
	public boolean isTaskRoot() {
		// TODO Auto-generated method stub
		return mBase.isTaskRoot();
	}

	@Override
	public boolean moveTaskToBack(boolean nonRoot) {
		// TODO Auto-generated method stub
		return mBase.moveTaskToBack(nonRoot);
	}

	@Override
	public String getLocalClassName() {
		// TODO Auto-generated method stub
		return mBase.getLocalClassName();
	}

	@Override
	public ComponentName getComponentName() {
		// TODO Auto-generated method stub
		return mBase.getComponentName();
	}

	@Override
	public SharedPreferences getPreferences(int mode) {
		// TODO Auto-generated method stub
		return mBase.getPreferences(mode);
	}

	@Override
	public Object getSystemService(String name) {
		// TODO Auto-generated method stub
		return mBase.getSystemService(name);
	}

	@Override
	public void setTitle(CharSequence title) {
		// TODO Auto-generated method stub
		mBase.setTitle(title);
	}

	@Override
	public void setTitle(int titleId) {
		// TODO Auto-generated method stub
		mBase.setTitle(titleId);
	}

	@Override
	public void setTitleColor(int textColor) {
		// TODO Auto-generated method stub
		mBase.setTitleColor(textColor);
	}


	@Override
	public View onCreateView(String name, Context context, AttributeSet attrs) {
		// TODO Auto-generated method stub
		return mBase.onCreateView(name, context, attrs);
	}

	@Override
	public View onCreateView(View parent, String name, Context context,
			AttributeSet attrs) {
		// TODO Auto-generated method stub
		return mBase.onCreateView(parent, name, context, attrs);
	}

	@Override
	public void dump(String prefix, FileDescriptor fd, PrintWriter writer,
			String[] args) {
		// TODO Auto-generated method stub
		mBase.dump(prefix, fd, writer, args);
	}

	@Override
	public ActionMode startActionMode(Callback callback) {
		// TODO Auto-generated method stub
		return mBase.startActionMode(callback);
	}

	@Override
	public ActionMode onWindowStartingActionMode(Callback callback) {
		// TODO Auto-generated method stub
		return mBase.onWindowStartingActionMode(callback);
	}

	@Override
	public void onActionModeStarted(ActionMode mode) {
		// TODO Auto-generated method stub
		mBase.onActionModeStarted(mode);
	}

	@Override
	public void onActionModeFinished(ActionMode mode) {
		// TODO Auto-generated method stub
		mBase.onActionModeFinished(mode);
	}

	@Override
	public boolean shouldUpRecreateTask(Intent targetIntent) {
		// TODO Auto-generated method stub
		return mBase.shouldUpRecreateTask(targetIntent);
	}

	@Override
	public boolean navigateUpTo(Intent upIntent) {
		// TODO Auto-generated method stub
		return mBase.navigateUpTo(upIntent);
	}

	@Override
	public boolean navigateUpToFromChild(Activity child, Intent upIntent) {
		// TODO Auto-generated method stub
		return mBase.navigateUpToFromChild(child, upIntent);
	}

	@Override
	public Intent getParentActivityIntent() {
		// TODO Auto-generated method stub
		return mBase.getParentActivityIntent();
	}
	
}
