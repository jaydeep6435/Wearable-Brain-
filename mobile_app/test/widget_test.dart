
import 'package:flutter_test/flutter_test.dart';
import 'package:memory_assistant/main.dart';

void main() {
  testWidgets('App renders bottom navigation', (WidgetTester tester) async {
    await tester.pumpWidget(const MemoryAssistantApp());

    // Verify bottom navigation tabs exist
    expect(find.text('Home'), findsOneWidget);
    expect(find.text('Query'), findsOneWidget);
    expect(find.text('Reminders'), findsOneWidget);
  });
}
